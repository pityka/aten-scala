
#include <jni.h>       
#include <iostream>     
#include <exception>  
#include <stdlib.h>
#include <string.h>
#include <ATen/Functions.h>
using namespace std;
using namespace at;

jlongArray vecToJni(JNIEnv *env, std::vector<int64_t> vec){
    
   int len = vec.size();
  jlongArray ret = env->NewLongArray( len);
  int64_t* buf = vec.data();
 
  env->SetLongArrayRegion(ret,0,len,(jlong*)buf);
  
   return ret;

}

jobject allocateTensor(JNIEnv *env, Tensor tensor) {
  jclass cls2 = env->FindClass("aten/Tensor");
  jmethodID mid = env->GetMethodID( cls2, "<init>", "(J)V");
  Tensor* result_on_heapreturnable_result = new Tensor(tensor);
  jlong addr = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  jobject ret_obj = env->NewObject( cls2, mid, addr);
  return ret_obj;
}
jobject allocateTensorOptions(JNIEnv *env, TensorOptions* tensorOptions) {
  jclass cls2 = env->FindClass("aten/TensorOptions");
  jmethodID mid = env->GetMethodID( cls2, "<init>", "(J)V");
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
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda_index(JNIEnv *env, jobject thisObj, jint index) {
    try {
      jclass cls = env->GetObjectClass( thisObj);
      TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
      
      TensorOptions* t2 = new TensorOptions(tensorOptions->device_index(index));
      return allocateTensorOptions(env,t2);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device(at::kCUDA));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cpu(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device(at::kCPU));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toDouble(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->dtype(kDouble));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toLong(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->dtype(kLong));
     return allocateTensorOptions(env,t2);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toFloat(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    TensorOptions* opt = new TensorOptions(tensor.options());

    return allocateTensorOptions(env,opt);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jint JNICALL Java_aten_Tensor_dim(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.dim();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_defined(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.defined();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_isCuda(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.is_cuda();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_useCount(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.weak_use_count();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_weakUseCount(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.use_count();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_numel(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.numel();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_elementSize(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.element_size();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jbyte JNICALL Java_aten_Tensor_scalarType(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return static_cast<jbyte>(tensor.scalar_type());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jstring JNICALL Java_aten_Tensor_nativeToString(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return env->NewStringUTF(tensor.toString().c_str());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_sizes(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    std::vector<int64_t> s = tensor.sizes().vec();
    return vecToJni(env,s);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_strides(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    std::vector<int64_t> s = tensor.strides().vec();
    return vecToJni(env,s);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_setToTensor(JNIEnv *env, jobject thisObj, jobject other) {
    try{
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    Tensor otherTensor = *reinterpret_cast<Tensor*>(env->GetLongField( other, env->GetFieldID( cls, "pointer", "J")));
    tensor = otherTensor;
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_print(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    tensor.print();
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_mul_1(JNIEnv *env, jobject thisObj, jdouble d) {try{
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    tensor.mul_(d);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_add_1(JNIEnv *env, jobject thisObj, jdouble other, jdouble alpha) {try{
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    tensor.add_(other,alpha);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_release(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    delete tensor;

    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromFloatArray(JNIEnv *env, jobject thisObj, jfloatArray datain) {try{
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
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
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    Tensor retTensor = tensor.cpu();
    
      return allocateTensor(env,retTensor);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  
  JNIEXPORT jobject JNICALL Java_aten_Tensor_cuda(JNIEnv *env, jobject thisObj) {
    try{
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    Tensor retTensor = tensor.cuda();
    
    return allocateTensor(env,retTensor);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
}