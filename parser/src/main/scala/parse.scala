import fastparse._, NoWhitespace._
import java.awt.RenderingHints.Key

object Parser extends App {
  val source = scala.io.Source.fromFile(args(0)).mkString

  case class TpeData(
      tpe: String,
      pointer: Option[String],
      members: Seq[TpeData]
  )
  case class ArgData(tpe: TpeData, name: String)
  case class DeclData(
      retTpe: TpeData,
      fnName: String,
      args: Seq[ArgData],
      generatedFnName: String
  )

  def WSChars[_: P] = P(NoTrace(CharsWhileIn("\u0020\u0009")))

  def SameLineCharChunks[_: P] =
    P(CharsWhile(c => c != '\n' && c != '\r') | !Newline ~ AnyChar)

  def LineComment[_: P] =
    P("//" ~ SameLineCharChunks.rep ~ Newline.rep(1))
  def WS[_: P]: P[Unit] = P((WSChars).rep)
  def WSNL[_: P]: P[Unit] = P((WSChars | StringIn("\r\n", "\n")).rep)
  def WSOrAmp[_: P]: P[Unit] = P((CharIn("\u0020\u0009\\&")).rep)
  def Newline[_: P] = P(NoTrace(StringIn("\r\n", "\n")))
  def Semi[_: P] = P(";" | Newline.rep(1))
  def Alphanumeric[_: P] =
    P(
      CharPred((c: Char) =>
        CharPredicates.isLetter(c) | CharPredicates
          .isDigit(c) | c == '$' | c == '_'
      )
    )

  def TypeName[_: P]: P[TpeData] =
    P(
      (Alphanumeric | ":")
        .rep(1)
        .! ~ ("<" ~ WS ~ TypeName
        .rep(sep = ",") ~ WS ~ ">").? ~ (WS ~ ("*" | "&")).!.?
    ).map {
      case (name, members, pointer) =>
        TpeData(name, pointer.map(_.trim), members.toSeq.flatten)
    }
  // (Alphanumeric | CharIn(":<>") | "," | "&").rep(1)
  def AnyButNewLine[_: P] = P(CharsWhile(c => c != '\r' && c != '\n'))

  def EmptyLine[_: P] = P(WS ~ Newline.rep(1))

  def Pragma[_: P] =
    P("#") ~ Alphanumeric.rep ~ WS ~ AnyButNewLine ~ Newline.rep(1)
  def Using[_: P] =
    P("using") ~ WS ~ AnyButNewLine ~ Newline.rep(1)

  def Namespace[_: P] =
    P(
      "namespace" ~ WS ~ Alphanumeric
        .rep(1) ~ WS ~ "{" ~/ WS ~ Newline.rep ~/ NamespaceBody ~ WS ~ Newline.rep ~ WS ~ "}" ~ WS ~ Newline
        .rep(1)
    )

  def NamespaceBody[_: P] =
    P(
      (EmptyLine | LineComment | Using | Decl).rep
    ).map { _ collect { case s: DeclData => s } }

  def ArrayLiteral[_: P] =
    P("{" ~ (WS ~ Alphanumeric.rep(1) ~ WS).rep(sep = ",") ~ "}")

  def Arg[_: P] =
    P(
      WSNL ~ Keywords ~ WSNL ~ TypeName
        ~ WSNL ~ Alphanumeric
        .rep(1)
        .! ~ ("=" ~ (":" | "-" | "." | ArrayLiteral | Alphanumeric | CharIn(
        "{}"
      )).rep(1)).?
    ).map { case (tpe, name) => ArgData(tpe, name) }

  def Decl[_: P] =
    P(
      Keywords ~ WSNL ~ TypeName ~ WSNL
        ~ Alphanumeric.rep(1).! ~/ "(" ~/ WSNL ~ Arg.rep(sep = ",")
        ~ WSNL ~ ")" ~ WS ~ Semi
    ).map {
      case (retTpe, fnName, args) =>
        DeclData(retTpe, fnName, args, "")
    }

  def Keyword[_: P] = P(StringIn("static", "inline", "const", "&"))

  def Keywords[_: P] = Keyword.rep(sep = WSChars.rep(1))

  def Doc[_: P] =
    P(
      Start ~ (Newline | EmptyLine | Pragma | LineComment).rep ~ Namespace ~ End
    )

  val parsed = parse(source, Doc(_)).get.value
    .groupBy(decl =>
      (
        decl.fnName
      )
    )
    .toSeq
    .flatMap {
      case (key, group) =>
        if (group.size > 1)
          group.zipWithIndex.map {
            case (decl, idx) =>
              decl.copy(generatedFnName = decl.fnName + "_" + idx.toString)
          }
        else group.map(decl => decl.copy(generatedFnName = decl.fnName))
    }

  val returnTypes = parsed.map(d => d.retTpe).distinct

  val argTypes = parsed.flatMap(d => d.args.map(a => a.tpe)).distinct

  println("returns: ")
  println(returnTypes.mkString("\n"))

  println("args: ")
  println(argTypes.mkString("\n"))

  val packageAndClassName = "aten_ATen"

  case class MappedType(
      argName: String,
      cType: ArgData,
      jniArgument: String,
      convertFromJni: String,
      javaType: String,
      noInputFromJava: Boolean = false
  )

  case class MappedReturnType(
      convert: String,
      jniType: String,
      javaType: String,
      cType: String
  ) {
    def shortJavaType = javaType match {
      case "long"   => "J"
      case "double" => "D"
      case "int"    => "I"
      case _        => "O"
    }
  }

  def mapReturnType(
      cType: TpeData,
      argName: String,
      returnable: String,
      toplevel: Boolean
  ): MappedReturnType = {
    cType match {
      case TpeData("void", None, List()) =>
        MappedReturnType(
          convert = s"""""",
          jniType = "void",
          javaType = "void",
          cType = "void"
        )
      case TpeData("int64_t", None, List()) =>
        MappedReturnType(
          convert = s"""int64_t $returnable = $argName;""",
          jniType = "jlong",
          javaType = "long",
          cType = "int64_t"
        )
      case TpeData("double", None, List()) =>
        MappedReturnType(
          convert = s"""double $returnable = $argName;""",
          jniType = "jdouble",
          javaType = "double",
          cType = "double"
        )
      case TpeData("bool", None, List()) =>
        MappedReturnType(
          convert = s"""bool $returnable = $argName;""",
          jniType = "jboolean",
          javaType = "boolean",
          cType = "bool"
        )

      case TpeData(
          "std::tuple",
          None,
          members
          ) =>
        val mappedMembers = members.zipWithIndex.map {
          case (tpe, idx) =>
            mapReturnType(
              tpe,
              s"std::get<$idx>($argName)",
              returnable + "_" + idx,
              false
            )
        }
        MappedReturnType(
          convert = s"""
      ${mappedMembers.map(_.convert).mkString("\n\n")}
      jclass _cls$returnable = env->FindClass("scala/Tuple${members.size}");
      jmethodID _midInit$returnable = env->GetMethodID( _cls$returnable, "<init>", "(${mappedMembers
            .map(_ => "Ljava/lang/Object")
            .mkString("", ";", ";")})V");
  jobject ret_out$returnable = env->NewObject(_cls$returnable, _midInit$returnable, ${mappedMembers.zipWithIndex
            .map(v => returnable + "_" + v._2)
            .mkString(",")});
       jobject $returnable = ret_out$returnable;
       """,
          jniType = "jobject",
          javaType =
            s"scala.Tuple${members.size}<${mappedMembers.map(_.javaType).mkString(",")}>",
          cType = s"std::tuple<${mappedMembers.map(_.cType).mkString(",")}>"
        )
      case TpeData(
          "std::vector",
          None,
          List(TpeData("Tensor", None, List()))
          ) =>
        MappedReturnType(
          convert = s"""
  int ret_len$returnable = $argName.size();
  jclass ret_cls$returnable = env->FindClass("aten/Tensor");
  jmethodID ret_midInit$returnable = env->GetMethodID( ret_cls$returnable, "<init>", "(J)V");
  jobjectArray ret_out$returnable = env->NewObjectArray( ret_len$returnable, ret_cls$returnable, nullptr);
  for (int i = 0; i < ret_len$returnable;i++) {
    jlong ret_address = reinterpret_cast<jlong>(new Tensor($argName.at(i)));
    jobject ret_obj = env->NewObject( ret_cls$returnable, ret_midInit$returnable, ret_address);
    env->SetObjectArrayElement(ret_out$returnable, i, ret_obj);
  }
   jobject $returnable = ret_out$returnable;""",
          jniType = "jobject",
          javaType = "Tensor[]",
          cType = "std::vector<Tensor>"
        )
      case TpeData("Tensor", Some("&"), Nil) if !toplevel =>
        MappedReturnType(
          convert = s"""
  jclass ret_cls$returnable = env->FindClass("aten/Tensor");
  jmethodID ret_midInit$returnable = env->GetMethodID( ret_cls$returnable, "<init>", "(J)V");
  jlong ret_address$returnable = reinterpret_cast<jlong>(&$argName);
  jobject ret_obj$returnable = env->NewObject( ret_cls$returnable, ret_midInit$returnable, ret_address$returnable);
   jobject $returnable = ret_obj$returnable;""",
          jniType = "jobject",
          javaType = "Tensor",
          cType = "Tensor"
        )
      case TpeData("Tensor", Some("&"), Nil) =>
        MappedReturnType(
          convert = "",
          jniType = "void",
          javaType = "void",
          cType = "void"
        )
      case TpeData("ScalarType", None, List()) =>
        MappedReturnType(
          convert = s"""
   jbyte $returnable = static_cast<int8_t>($argName);""",
          jniType = "jbyte",
          javaType = "byte",
          cType = "ScalarType"
        )
      case TpeData("Tensor", None, Nil) =>
        MappedReturnType(
          convert = s"""
  jclass ret_cls$returnable = env->FindClass("aten/Tensor");
  jmethodID ret_midInit$returnable = env->GetMethodID( ret_cls$returnable, "<init>", "(J)V");
  Tensor* result_on_heap$returnable = new Tensor($argName);
  jlong ret_address$returnable = reinterpret_cast<jlong>(result_on_heap$returnable);
  jobject ret_obj$returnable = env->NewObject( ret_cls$returnable, ret_midInit$returnable, ret_address$returnable);
   jobject $returnable = ret_obj$returnable;""",
          jniType = "jobject",
          javaType = "Tensor",
          cType = "Tensor"
        )

    }
  }

  def mapType(cType: ArgData): MappedType = cType match {
    case arg @ ArgData(TpeData("Scalar", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = ""
      MappedType(
        jniArgName,
        arg,
        "jdouble " + jniArgName,
        convertFromJni,
        "double"
      )
    case arg @ ArgData(TpeData("double", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = ""
      MappedType(
        jniArgName,
        arg,
        "jdouble " + jniArgName,
        convertFromJni,
        "double"
      )
    case arg @ ArgData(TpeData("IntArrayRef", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = s"""
      jsize ${jniArgName}_length = env->GetArrayLength($jniArgName);
      int64_t* ${jniArgName}_ar = new int64_t[${jniArgName}_length];
      jlong* ${jniArgName}_ar2 = env->GetLongArrayElements($jniArgName,nullptr);
      for (int i = 0; i < ${jniArgName}_length; i++) {
          ${jniArgName}_ar[i] = ${jniArgName}_ar2[i];
      }
      env->ReleaseLongArrayElements($jniArgName,${jniArgName}_ar2,0);
      IntArrayRef ${jniArgName}_c = *(new IntArrayRef(${jniArgName}_ar,${jniArgName}_length));
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jlongArray " + jniArgName,
        convertFromJni,
        "long[]"
      )
    case arg @ ArgData(TpeData("TensorList", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = s"""
      jsize ${jniArgName}_length = env->GetArrayLength($jniArgName);
      Tensor* ${jniArgName}_ar = new Tensor[${jniArgName}_length];
      for (int i = 0; i < ${jniArgName}_length; i++) {
         jobject obj = env->GetObjectArrayElement( $jniArgName, i);
          jclass cls = env->GetObjectClass( obj);
        jfieldID fid = env->GetFieldID( cls, "pointer", "J");
         jlong address = env->GetLongField( obj, fid);
         Tensor* pointer = reinterpret_cast<Tensor*>(address);
          ${jniArgName}_ar[i] = *pointer;
      }
      TensorList ${jniArgName}_c = *(new TensorList(${jniArgName}_ar,${jniArgName}_length));
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jobjectArray " + jniArgName,
        convertFromJni,
        "Tensor[]"
      )
    case arg @ ArgData(TpeData("Generator", Some("*"), List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = ""
      MappedType(
        "nullptr",
        arg,
        "",
        convertFromJni,
        "",
        true
      )
    case arg @ ArgData(TpeData("TensorOptions", Some("&"), List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = s"""
   jclass ${jniArgName}_class = env->GetObjectClass( $jniArgName);
   jfieldID ${jniArgName}_fidNumber = env->GetFieldID( ${jniArgName}_class, "pointer", "J");
   jlong ${jniArgName}_pointer = env->GetLongField( $jniArgName, ${jniArgName}_fidNumber);
   TensorOptions ${jniArgName}_c = *reinterpret_cast<TensorOptions*>(${jniArgName}_pointer);
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jobject " + jniArgName,
        convertFromJni,
        "TensorOptions"
      )
    case arg @ ArgData(TpeData("c10::optional", None, _), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = ""
      MappedType(
        "c10::nullopt",
        arg,
        "",
        convertFromJni,
        "",
        true
      )
    case arg @ ArgData(TpeData("int64_t", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = ""
      MappedType(
        jniArgName,
        arg,
        "jlong " + jniArgName,
        convertFromJni,
        "long"
      )
    case arg @ ArgData(TpeData("Tensor", Some("&"), List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = s"""
   jclass ${jniArgName}_class = env->GetObjectClass( $jniArgName);
   jfieldID ${jniArgName}_fidNumber = env->GetFieldID( ${jniArgName}_class, "pointer", "J");
   jlong ${jniArgName}_pointer = env->GetLongField( $jniArgName, ${jniArgName}_fidNumber);
   Tensor ${jniArgName}_c = *reinterpret_cast<Tensor*>(${jniArgName}_pointer);
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jobject " + jniArgName,
        convertFromJni,
        "Tensor"
      )
    //   case arg @ ArgData(TpeData("Dimname", None, List()), argName) =>
    //     val jniArgName = "jniparam_" + argName
    //     val convertFromJni = s"""
    //  jclass ${jniArgName}_class = env->GetObjectClass( $jniArgName);
    //  jfieldID ${jniArgName}_fidNumber = env->GetFieldID( ${jniArgName}_class, "pointer", "J");
    //  jlong ${jniArgName}_pointer = env->GetLongField( $jniArgName, ${jniArgName}_fidNumber);
    //  Dimname ${jniArgName}_c = *reinterpret_cast<Dimname*>(${jniArgName}_pointer);
    //     """
    //     MappedType(
    //       jniArgName + "_c",
    //       arg,
    //       "jobject " + jniArgName,
    //       convertFromJni,
    //       "Dimname"
    //     )
    // case arg @ ArgData(TpeData("DimnameList", None, List()), argName) =>
    //   val jniArgName = "jniparam_" + argName
    //   val convertFromJni = s"""
    //   jsize ${jniArgName}_length = env->GetArrayLength($jniArgName);
    //   std::vector<Dimname> ${jniArgName}_ar;
    //   for (int i = 0; i < ${jniArgName}_length; i++) {
    //      jobject obj = env->GetObjectArrayElement( $jniArgName, i);
    //       jclass cls = env->GetObjectClass( obj);
    //     jfieldID fid = env->GetFieldID( cls, "pointer", "J");
    //      jlong address = env->GetLongField( obj, fid);
    //      Dimname* pointer = reinterpret_cast<Dimname*>(address);
    //       ${jniArgName}_ar.push_back(*pointer);
    //   }
    //   DimnameList ${jniArgName}_c = *(new DimnameList(${jniArgName}_ar.data(),${jniArgName}_length));
    //   """
    //   MappedType(
    //     jniArgName + "_c",
    //     arg,
    //     "jobjectArray " + jniArgName,
    //     convertFromJni,
    //     "Dimname[]"
    //   )
    case arg @ ArgData(TpeData("ScalarType", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni =
        s"""ScalarType ${jniArgName}_c  = static_cast<ScalarType>((int8_t)$jniArgName);
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jbyte " + jniArgName,
        convertFromJni,
        "byte"
      )
    case arg @ ArgData(TpeData("std::string", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni =
        s"""std::string ${jniArgName}_c  = jstring2string(env,$jniArgName);
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jstring " + jniArgName,
        convertFromJni,
        "String"
      )
    case arg @ ArgData(TpeData("bool", None, List()), argName) =>
      val jniArgName = "jniparam_" + argName
      val convertFromJni = ""
      MappedType(
        jniArgName,
        arg,
        "jboolean " + jniArgName,
        convertFromJni,
        "boolean"
      )
    case arg @ ArgData(
          TpeData(
            "std::array",
            None,
            List(TpeData("bool", None, List()), TpeData(numStr, None, List()))
          ),
          argName
        ) =>
      val num = numStr.toInt
      val jniArgName = "jniparam_" + argName
      val convertFromJni = s"""
      
      jboolean* ${jniArgName}_ar2 = env->GetBooleanArrayElements($jniArgName,nullptr);
      
      std::array<bool,$numStr> ${jniArgName}_c = {
        ${(1 to num)
        .map { i => s"(bool)${jniArgName}_ar2[$i]" }
        .mkString(",")}
        };
        env->ReleaseBooleanArrayElements($jniArgName,${jniArgName}_ar2,0);
      """
      MappedType(
        jniArgName + "_c",
        arg,
        "jbooleanArray " + jniArgName,
        convertFromJni,
        "boolean[]"
      )
  }

  def implementCpp(decl: DeclData) = {
    val mappedArgs = decl.args.map(mapType)
    val mappedRet =
      mapReturnType(decl.retTpe, "result", "returnable_result", true)
    val javaArgumentList = {
      val l = mappedArgs
        .filterNot(_.noInputFromJava)
        .map(_.jniArgument)

      if (l.nonEmpty)
        l.mkString(",", ",", "")
      else ""
    }
    s"""JNIEXPORT ${mappedRet.jniType} JNICALL Java_${packageAndClassName}_${decl.generatedFnName
      .replaceAllLiterally(
        "_",
        "_1"
      )}(JNIEnv *env, jobject thisObj $javaArgumentList) {
  ${mappedArgs.map(_.convertFromJni).mkString("\n")}

  ${if (mappedRet.cType == "void") "" else s"${mappedRet.cType} result = "} at::${decl.fnName}(${mappedArgs
      .map(_.argName)
      .mkString(",")});
  

   ${mappedRet.convert}
    ${if (mappedRet.cType == "void") "return;"
    else "return returnable_result;"}
}"""
  }
  def implementJava(decl: DeclData) = {
    val mappedArgs = decl.args.map(mapType)
    val mappedRet = mapReturnType(decl.retTpe, "", "", true)
    val javaArgumentList =
      mappedArgs
        .filterNot(_.noInputFromJava)
        .map(d => d.javaType + " " + d.cType.name)
        .mkString(",")
    s"""public static native ${mappedRet.javaType} ${decl.generatedFnName}($javaArgumentList);"""
  }

  val cpp = s"""
#include <jni.h>       
#include <iostream>       
#include <stdlib.h>
#include <string.h>
#include <ATen/Functions.h>
using namespace std;
using namespace at;

std::string jstring2string(JNIEnv *env, jstring jStr) {
    const jclass stringClass = env->GetObjectClass(jStr);
    const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
    const jbyteArray stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

    size_t length = (size_t) env->GetArrayLength(stringJbytes);
    jbyte* pBytes = env->GetByteArrayElements(stringJbytes, NULL);

    std::string ret = std::string((char *)pBytes, length);
    env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

    env->DeleteLocalRef(stringJbytes);
    env->DeleteLocalRef(stringClass);
    return ret;
}

extern "C" {

${parsed.sortBy(_.fnName).map(implementCpp).mkString("\n\n")}
}
"""
  val javaSrc = s"""
package aten ;

public class ATen {

  static {
    Load.load();
  }

${parsed.sortBy(_.fnName).map(implementJava).mkString("\n\n")}
}
"""

  println("Parsed OK")
  // println(parsed.mkString("\n"))
  // println(cpp)
  // println(javaSrc)
  if (args.size >= 2) {
    val out = args(1)
    val fw = new java.io.FileWriter(new java.io.File(out))
    fw.write(cpp)
    fw.close
  }
  if (args.size >= 3) {
    val out = args(2)
    val fw = new java.io.FileWriter(new java.io.File(out))
    fw.write(javaSrc)
    fw.close
  }
}
