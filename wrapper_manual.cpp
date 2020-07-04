
#include <jni.h>       
#include <iostream>     
#include <exception>  
#include <stdlib.h>
#include <string.h>
#include <ATen/Functions.h>
#include "wrapper_manual.h"
using namespace std;
using namespace at;

static jint JNI_VERSION = JNI_VERSION_1_1;

jclass tensorClass;
jmethodID tensorCtor;
jfieldID tensorPointerFid;

jclass tensorOptionsClass;
jmethodID tensorOptionsCtor;
jfieldID tensorOptionsPointerFid;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {

    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {

        cout << "jni abort" << endl;
        return JNI_ERR;
    }

    jclass tempLocalClassRef;
    jclass tempLocalClassRef2;
    tempLocalClassRef = env->FindClass("aten/Tensor");
    tensorClass = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);
    tensorCtor = env->GetMethodID( tensorClass, "<init>", "(J)V");
    tensorPointerFid = env->GetFieldID( tensorClass, "pointer", "J");

    tempLocalClassRef = env->FindClass("aten/TensorOptions");
    tensorOptionsClass = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);
    tensorOptionsCtor = env->GetMethodID( tensorOptionsClass, "<init>", "(J)V");
    tensorOptionsPointerFid = env->GetFieldID( tensorOptionsClass, "pointer", "J");

    return JNI_VERSION;
}

void JNI_OnUnload(JavaVM *vm, void *reserved) {

    JNIEnv* env;
    vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION);

    env->DeleteGlobalRef(tensorClass);
    env->DeleteGlobalRef(tensorOptionsClass);

}

jlongArray vecToJni(JNIEnv *env, std::vector<int64_t> vec){
    
   int len = vec.size();
  jlongArray ret = env->NewLongArray( len);
  int64_t* buf = vec.data();
 
  env->SetLongArrayRegion(ret,0,len,(jlong*)buf);
  
   return ret;

}

jobject allocateTensor(JNIEnv *env, Tensor tensor) {
  Tensor* result_on_heapreturnable_result = new Tensor(tensor);
  jlong addr = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_obj = env->NewObject( tensorClass, tensorCtor, addr);
  return ret_obj;
}
jobject allocateTensorOptions(JNIEnv *env, TensorOptions* tensorOptions) {
  jclass cls2 = tensorOptionsClass;
  jmethodID mid = tensorOptionsCtor;
  jlong addr = reinterpret_cast<jlong>(tensorOptions);
  jobject ret_obj = env->NewObject( cls2, mid, addr);
  return ret_obj;
}

jint throwRuntimeException( JNIEnv *env, const char *message )
{

    jclass exClass = env->FindClass(  "java/lang/RuntimeException" );
    if ( exClass == NULL ) {
        return -1;
    }

    return env->ThrowNew(  exClass, message );
}

extern "C" {
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda_1index(JNIEnv *env, jobject thisObj, jint index) {
    try {
      jclass cls = tensorOptionsClass;
      TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
      
      TensorOptions* t2 = new TensorOptions(tensorOptions->device_index(index));
      return allocateTensorOptions(env,t2);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device(at::kCUDA));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cpu(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device(at::kCPU));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jbyte JNICALL Java_aten_Tensor_scalarTypeByte(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
    int8_t tpe = (int8_t)c10::typeMetaToScalarType(tensor->options().dtype());
     return tpe;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jbyte JNICALL Java_aten_TensorOptions_scalarTypeByte(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    int8_t tpe = (int8_t)c10::typeMetaToScalarType(tensorOptions->dtype());
     return tpe;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_TensorOptions_isCPU(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    bool tpe = tensorOptions->device().is_cpu();
     return tpe;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_TensorOptions_isCuda(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    bool tpe = tensorOptions->device().is_cuda();
     return tpe;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_cudnnAvailable(JNIEnv *env, jobject thisObj) { try{
    
    bool ret = at::detail::getCUDAHooks().getNumGPUs() > 0 && at::detail::getCUDAHooks().hasCuDNN();
    return ret;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jshort JNICALL Java_aten_TensorOptions_deviceIndex(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    bool tpe = tensorOptions->device().index();
     return tpe;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toDouble(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->dtype(kDouble));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toLong(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->dtype(kLong));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toFloat(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->dtype(kFloat));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_dtypeFloat(JNIEnv *env, jobject thisObj) {try{
    
    TensorOptions* tensorOptions =new TensorOptions((ScalarType)6);
    
     return allocateTensorOptions(env,tensorOptions);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_dtypeLong(JNIEnv *env, jobject thisObj) {try{
    
    TensorOptions* tensorOptions =new TensorOptions((ScalarType)4);
    
     return allocateTensorOptions(env,tensorOptions);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_dtypeDouble(JNIEnv *env, jobject thisObj) {
    try{
    TensorOptions* tensorOptions =new TensorOptions((ScalarType)7);
    
     return allocateTensorOptions(env,tensorOptions);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }

  JNIEXPORT jobject JNICALL Java_aten_Tensor_options(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    TensorOptions* opt = new TensorOptions(tensor.options());

    return allocateTensorOptions(env,opt);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jint JNICALL Java_aten_Tensor_dim(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.dim();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_defined(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.defined();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_isCuda(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.is_cuda();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_useCount(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.weak_use_count();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_weakUseCount(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.use_count();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_numel(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.numel();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_elementSize(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor.element_size();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jbyte JNICALL Java_aten_Tensor_scalarType(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return static_cast<jbyte>(tensor.scalar_type());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jstring JNICALL Java_aten_Tensor_nativeToString(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return env->NewStringUTF(tensor.toString().c_str());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_sizes(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    std::vector<int64_t> s = tensor.sizes().vec();
    return vecToJni(env,s);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_strides(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    std::vector<int64_t> s = tensor.strides().vec();
    return vecToJni(env,s);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_setToTensor(JNIEnv *env, jobject thisObj, jobject other) {
    try{
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor otherTensor = *reinterpret_cast<Tensor*>(env->GetLongField( other, tensorPointerFid));
    tensor = otherTensor;
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_print(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor.print();
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_mul_1(JNIEnv *env, jobject thisObj, jdouble d) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor.mul_(d);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_add_1(JNIEnv *env, jobject thisObj, jdouble other, jdouble alpha) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor.add_(other,alpha);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_release(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    delete tensor;

    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromFloatArray(JNIEnv *env, jobject thisObj, jfloatArray datain) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 6 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      auto cpuTensor = tensor.cpu().flatten();
      auto accessor = cpuTensor.accessor<float,1>();
      int64_t size = accessor.size(0);
      for (int64_t i = 0;i < size;i++){
        accessor[i] = in[i];
      }
      env->ReleaseFloatArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyToFloatArray(JNIEnv *env, jobject thisObj, jfloatArray datain) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 6 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      auto cpuTensor = tensor.cpu().flatten();
      auto accessor = cpuTensor.accessor<float,1>();
      int64_t size =  accessor.size(0);
      for (int64_t i = 0;i < size;i++){
        in[i] = accessor[i];
      }
      env->ReleaseFloatArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }

 JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromLongArray(JNIEnv *env, jobject thisObj, jlongArray datain) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 4 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      jlong* in = env->GetLongArrayElements(datain, nullptr);
      auto cpuTensor = tensor.cpu().flatten();
      auto accessor = cpuTensor.accessor<int64_t,1>();
      int64_t size = accessor.size(0);
      for (int64_t i = 0;i < size;i++){
        accessor[i] = in[i];
      }
      env->ReleaseLongArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyToLongArray(JNIEnv *env, jobject thisObj, jlongArray datain) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 4 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      jlong* in = env->GetLongArrayElements(datain, nullptr);
      auto cpuTensor = tensor.cpu().flatten();
      auto accessor = cpuTensor.accessor<int64_t,1>();
      int64_t size = accessor.size(0);
      for (int64_t i = 0;i < size;i++){
        in[i] = accessor[i];
      }
      env->ReleaseLongArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }

  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromDoubleArray(JNIEnv *env, jobject thisObj, jdoubleArray datain) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 7 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      auto cpuTensor = tensor.cpu().flatten();
      auto accessor = cpuTensor.accessor<double,1>();
      int64_t size = accessor.size(0);
      for (int64_t i = 0;i < size;i++){
        accessor[i] = in[i];
      }
      env->ReleaseDoubleArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyToDoubleArray(JNIEnv *env, jobject thisObj, jdoubleArray datain) {try{
    
    jclass cls =tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 7 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      auto cpuTensor = tensor.cpu().flatten();
      auto accessor = cpuTensor.accessor<double,1>();
      int64_t size =  accessor.size(0);
      for (int64_t i = 0;i < size;i++){
        in[i] = accessor[i];
      }
      env->ReleaseDoubleArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jobject JNICALL Java_aten_Tensor_cpu(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor retTensor = tensor.cpu();
    
      return allocateTensor(env,retTensor);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  
  JNIEXPORT jobject JNICALL Java_aten_Tensor_cuda(JNIEnv *env, jobject thisObj) {
    try{
    
    jclass cls = tensorClass;
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor retTensor = tensor.cuda();
    
    return allocateTensor(env,retTensor);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }

  JNIEXPORT void JNICALL Java_aten_Tensor_releaseAll(JNIEnv *env, jobject thisObj ,jobjectArray jniparam_tensors) {try{
  
      jsize jniparam_tensors_length = env->GetArrayLength(jniparam_tensors);
      for (int i = 0; i < jniparam_tensors_length; i++) {
         jobject obj = env->GetObjectArrayElement( jniparam_tensors, i);
          jclass cls = tensorClass;
        jfieldID fid = tensorPointerFid;
         jlong address = env->GetLongField( obj, fid);
         Tensor* pointer = reinterpret_cast<Tensor*>(address);
          delete pointer;
      }


      
    return ;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
}

JNIEXPORT jobject JNICALL Java_aten_Tensor_transpose(JNIEnv *env, jobject thisObj ,jlong dim0, jlong dim1) {try{
  

   
   jclass tensor_cls = tensorClass;
   Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
      
  Tensor result =  tensor.transpose(dim0,dim1).contiguous();
  

   
  jclass ret_clsreturnable_result = tensorClass;
  jmethodID ret_midInitreturnable_result = tensorCtor;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_objreturnable_result = env->NewObject( ret_clsreturnable_result, ret_midInitreturnable_result, ret_addressreturnable_result);
   jobject returnable_result = ret_objreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
}

JNIEXPORT jobject JNICALL Java_aten_Tensor_to(JNIEnv *env, jobject thisObj ,jobject jniparam_options, jboolean jniparam_copy) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions jniparam_options_c = *reinterpret_cast<TensorOptions*>(jniparam_options_pointer);

   jclass tensor_cls = tensorClass;
   Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
      
  Tensor result =  tensor.to(jniparam_options_c,false,jniparam_copy);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  jmethodID ret_midInitreturnable_result = tensorCtor;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_objreturnable_result = env->NewObject( ret_clsreturnable_result, ret_midInitreturnable_result, ret_addressreturnable_result);
   jobject returnable_result = ret_objreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
}

JNIEXPORT jobject JNICALL Java_aten_Tensor_scalarDouble(JNIEnv *env, jobject thisObj ,jdouble jniparam_s,jobject jniparam_options) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions jniparam_options_c = *reinterpret_cast<TensorOptions*>(jniparam_options_pointer);
      

  Tensor result =  at::scalar_tensor(jniparam_s,jniparam_options_c);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  jmethodID ret_midInitreturnable_result = tensorCtor;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_objreturnable_result = env->NewObject( ret_clsreturnable_result, ret_midInitreturnable_result, ret_addressreturnable_result);
   jobject returnable_result = ret_objreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
}
JNIEXPORT jobject JNICALL Java_aten_Tensor_scalarLong(JNIEnv *env, jobject thisObj ,jlong jniparam_s,jobject jniparam_options) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions jniparam_options_c = *reinterpret_cast<TensorOptions*>(jniparam_options_pointer);
      
  int64_t p = (int64_t)jniparam_s;
  Tensor result =  at::scalar_tensor(p,jniparam_options_c);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  jmethodID ret_midInitreturnable_result = tensorCtor;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_objreturnable_result = env->NewObject( ret_clsreturnable_result, ret_midInitreturnable_result, ret_addressreturnable_result);
   jobject returnable_result = ret_objreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
}
JNIEXPORT jobject JNICALL Java_aten_Tensor_scalarFloat(JNIEnv *env, jobject thisObj ,jfloat jniparam_s,jobject jniparam_options) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions jniparam_options_c = *reinterpret_cast<TensorOptions*>(jniparam_options_pointer);
      

  Tensor result =  at::scalar_tensor(jniparam_s,jniparam_options_c);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  jmethodID ret_midInitreturnable_result = tensorCtor;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_objreturnable_result = env->NewObject( ret_clsreturnable_result, ret_midInitreturnable_result, ret_addressreturnable_result);
   jobject returnable_result = ret_objreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
}

JNIEXPORT void JNICALL Java_aten_Tensor_addmm_1out_1transposed2(JNIEnv *env, jobject thisObj ,jobject jniparam_out,jobject jniparam_self,jobject jniparam_mat1,jobject jniparam_mat2,jdouble jniparam_beta,jdouble jniparam_alpha) {try{
  
   jclass jniparam_out_class = tensorClass;
   jfieldID jniparam_out_fidNumber = tensorPointerFid;
   jlong jniparam_out_pointer = env->GetLongField( jniparam_out, jniparam_out_fidNumber);
   Tensor jniparam_out_c = *reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor jniparam_self_c = *reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor jniparam_mat1_c = *reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor jniparam_mat2_c = *reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::addmm_out(jniparam_out_c,jniparam_self_c,jniparam_mat1_c,jniparam_mat2_c.t(),jniparam_beta,jniparam_alpha);
  
      






   
    return;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
}

JNIEXPORT void JNICALL Java_aten_Tensor_addmm_1out_1transposed1(JNIEnv *env, jobject thisObj ,jobject jniparam_out,jobject jniparam_self,jobject jniparam_mat1,jobject jniparam_mat2,jdouble jniparam_beta,jdouble jniparam_alpha) {try{
  
   jclass jniparam_out_class = tensorClass;
   jfieldID jniparam_out_fidNumber = tensorPointerFid;
   jlong jniparam_out_pointer = env->GetLongField( jniparam_out, jniparam_out_fidNumber);
   Tensor jniparam_out_c = *reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor jniparam_self_c = *reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor jniparam_mat1_c = *reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor jniparam_mat2_c = *reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::addmm_out(jniparam_out_c,jniparam_self_c,jniparam_mat1_c.t(),jniparam_mat2_c,jniparam_beta,jniparam_alpha);
  
      






   
    return;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
}

JNIEXPORT void JNICALL Java_aten_Tensor_baddbmm_1out_1transposed2(JNIEnv *env, jobject thisObj ,jobject jniparam_out,jobject jniparam_self,jobject jniparam_mat1,jobject jniparam_mat2,jdouble jniparam_beta,jdouble jniparam_alpha) {try{
  
   jclass jniparam_out_class = tensorClass;
   jfieldID jniparam_out_fidNumber = tensorPointerFid;
   jlong jniparam_out_pointer = env->GetLongField( jniparam_out, jniparam_out_fidNumber);
   Tensor jniparam_out_c = *reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor jniparam_self_c = *reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor jniparam_mat1_c = *reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor jniparam_mat2_c = *reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::baddbmm_out(jniparam_out_c,jniparam_self_c,jniparam_mat1_c,jniparam_mat2_c.transpose(1,2),jniparam_beta,jniparam_alpha);
  
      






   
    return;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
}
JNIEXPORT void JNICALL Java_aten_Tensor_baddbmm_1out_1transposed1(JNIEnv *env, jobject thisObj ,jobject jniparam_out,jobject jniparam_self,jobject jniparam_mat1,jobject jniparam_mat2,jdouble jniparam_beta,jdouble jniparam_alpha) {try{
  
   jclass jniparam_out_class = tensorClass;
   jfieldID jniparam_out_fidNumber = tensorPointerFid;
   jlong jniparam_out_pointer = env->GetLongField( jniparam_out, jniparam_out_fidNumber);
   Tensor jniparam_out_c = *reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor jniparam_self_c = *reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor jniparam_mat1_c = *reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor jniparam_mat2_c = *reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::baddbmm_out(jniparam_out_c,jniparam_self_c,jniparam_mat1_c.transpose(1,2),jniparam_mat2_c,jniparam_beta,jniparam_alpha);
  
      






   
    return;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
}
}