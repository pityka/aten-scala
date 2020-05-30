
#include <jni.h>       
#include <iostream>       
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

extern "C" {
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda_index(JNIEnv *env, jobject thisObj, jint index) {
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device_index(index));
     return allocateTensorOptions(env,t2);
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device(at::kCUDA));
     return allocateTensorOptions(env,t2);
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cpu(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->device(at::kCPU));
     return allocateTensorOptions(env,t2);
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_dtypeFloat(JNIEnv *env, jobject thisObj) {
    
    TensorOptions* tensorOptions =new TensorOptions((ScalarType)6);
    
     return allocateTensorOptions(env,tensorOptions);
    
  }
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_dtypeDouble(JNIEnv *env, jobject thisObj) {
    
    TensorOptions* tensorOptions =new TensorOptions((ScalarType)7);
    
     return allocateTensorOptions(env,tensorOptions);
    
  }

  JNIEXPORT jint JNICALL Java_aten_Tensor_dim(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.dim();
    
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_defined(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.defined();
    
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_isCuda(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.is_cuda();
    
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_useCount(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.weak_use_count();
    
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_weakUseCount(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.use_count();
    
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_numel(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.numel();
    
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_elementSize(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return tensor.element_size();
    
  }
  JNIEXPORT jbyte JNICALL Java_aten_Tensor_scalarType(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return static_cast<jbyte>(tensor.scalar_type());
    
  }
  JNIEXPORT jstring JNICALL Java_aten_Tensor_nativeToString(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    return env->NewStringUTF(tensor.toString().c_str());
    
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_sizes(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    std::vector<int64_t> s = tensor.sizes().vec();
    return vecToJni(env,s);
    
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_strides(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    std::vector<int64_t> s = tensor.strides().vec();
    return vecToJni(env,s);
    
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_setToTensor(JNIEnv *env, jobject thisObj, jobject other) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    Tensor otherTensor = *reinterpret_cast<Tensor*>(env->GetLongField( other, env->GetFieldID( cls, "pointer", "J")));
    tensor = otherTensor;
    return;
    
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_print(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    tensor.print();
    return;
    
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_release(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    delete tensor;

    return;
    
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromFloatArray(JNIEnv *env, jobject thisObj, jfloatArray datain) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 6 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      float* data = (float*)tensor.data_ptr();
      for (int i = 0;i < len;i++){
        data[i] = in[i];
      }
      env->ReleaseFloatArrayElements(datain,in,0);
      return true;
    }
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyToFloatArray(JNIEnv *env, jobject thisObj, jfloatArray datain) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 6 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      float* data = (float*)tensor.data_ptr();
      for (int i = 0;i < len;i++){
        in[i] = data[i];
      }
      env->ReleaseFloatArrayElements(datain,in,0);
      return true;
    }
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromDoubleArray(JNIEnv *env, jobject thisObj, jdoubleArray datain) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 7 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      double* data = (double*)tensor.data_ptr();
      for (int i = 0;i < len;i++){
        data[i] = in[i];
      }
      env->ReleaseDoubleArrayElements(datain,in,0);
      return true;
    }
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyToDoubleArray(JNIEnv *env, jobject thisObj, jdoubleArray datain) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 7 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      double* data = (double*)tensor.data_ptr();
      for (int i = 0;i < len;i++){
        in[i] = data[i];
      }
      env->ReleaseDoubleArrayElements(datain,in,0);
      return true;
    }
  }
  JNIEXPORT jobject JNICALL Java_aten_Tensor_cpu(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    Tensor retTensor = tensor.cpu();
    
      return allocateTensor(env,retTensor);
    
  }
  JNIEXPORT jobject JNICALL Java_aten_Tensor_cuda(JNIEnv *env, jobject thisObj) {
    
    jclass cls = env->GetObjectClass( thisObj);
    Tensor tensor = *reinterpret_cast<Tensor*>(env->GetLongField( thisObj, env->GetFieldID( cls, "pointer", "J")));
    Tensor retTensor = tensor.cuda();
    
    return allocateTensor(env,retTensor);
    
  }
}