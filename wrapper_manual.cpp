
#include <jni.h>       
#include <iostream>     
#include <exception>  
#include <stdlib.h>
#include <string.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <c10/core/CPUAllocator.h>
#include "wrapper_manual.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;
using namespace at;

static jint JNI_VERSION = JNI_VERSION_1_1;

jclass tensorClass;
jfieldID tensorPointerFid;

jclass tensorOptionsClass;
jmethodID tensorOptionsCtor;
jfieldID tensorOptionsPointerFid;

jclass longClass;
jmethodID longCtor;

int64_t reinterpret_unsigned_to_signed(uint64_t x) {
  int64_t tmp;
  std::memcpy(&tmp, &x, sizeof(tmp));
  const int64_t y = tmp;
  return y;
}
uint64_t reinterpret_signed_to_unsigned(int64_t x) {
  uint64_t tmp;
  std::memcpy(&tmp, &x, sizeof(tmp));
  const uint64_t y = tmp;
  return y;
}

jint JNI_OnLoad(JavaVM* vm, void* reserved) {

    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {

        cout << "jni abort" << endl;
        return JNI_ERR;
    }

    jclass tempLocalClassRef;
    tempLocalClassRef = env->FindClass("aten/Tensor");
    tensorClass = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);
    tensorPointerFid = env->GetFieldID( tensorClass, "pointer", "J");

    tempLocalClassRef = env->FindClass("aten/TensorOptions");
    tensorOptionsClass = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);
    tensorOptionsCtor = env->GetMethodID( tensorOptionsClass, "<init>", "(J)V");
    tensorOptionsPointerFid = env->GetFieldID( tensorOptionsClass, "pointer", "J");

    tempLocalClassRef = env->FindClass("java/lang/Long");
    longClass = (jclass) env->NewGlobalRef(tempLocalClassRef);
    env->DeleteLocalRef(tempLocalClassRef);
    longCtor = env->GetMethodID( longClass, "<init>", "(J)V");

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

jlong allocateTensor(JNIEnv *env, Tensor tensor) {
  Tensor* result_on_heapreturnable_result = new Tensor(tensor);
  jlong addr = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  
  return addr;
}
jobject allocateTensorOptions(JNIEnv *env, TensorOptions* tensorOptions) {
  jlong addr = reinterpret_cast<jlong>(tensorOptions);
  jobject ret_obj = env->NewObject( tensorOptionsClass, tensorOptionsCtor, addr);
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

// Offset mmap


class OffsettableMMap {
 public:
  OffsettableMMap(const char *filename, size_t size, size_t offset, bool pin);
  
  void close();
  static at::DataPtr makeDataPtr(const char *filename, size_t size, size_t offset, bool pin);

  void* pointer() const { return base_ptr_; }
  ~OffsettableMMap() {
    close();
    c10::reportMemoryUsageToProfiler(base_ptr_, -size_, c10::Device(c10::DeviceType::CPU));
  }

private:
  bool closed_ = false;
  bool pinned_ = false;
  ptrdiff_t size_; 
  void *base_ptr_ = nullptr;
};


OffsettableMMap::OffsettableMMap( const char *filename,  size_t size, size_t offset, bool pin)
  : size_(size) // to be filled later
  , base_ptr_(nullptr)
{

  if (size == 0) {
     throw std::runtime_error("trying to mmap non positive size");
  }
  int fd;
  struct stat file_stat;
  pinned_ = pin;

  if ((fd = open(filename, O_RDONLY)) == -1) {
    AT_ERROR("unable to open file <", filename, "> in read-only mode");
  }

  if (fstat(fd, &file_stat) == -1) {
    ::close(fd);
    AT_ERROR("unable to stat the file <", filename, ">");
  }

  if (size  > file_stat.st_size-offset) {
    ::close(fd);
    throw std::runtime_error("trying to mmap more than file size");
  }
  
  base_ptr_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, offset);

  if (pin) {
    if(mlock(base_ptr_,size_)) {
      AT_ERROR("Error locking page: ", strerror(errno), " (", errno, ")");
    }
  }
   
  if (base_ptr_ == MAP_FAILED) {
    ::close(fd);
    base_ptr_ = nullptr; /* let's be sure it is NULL */
    AT_ERROR("unable to mmap ", size_, " bytes from file <", filename, ">: ", strerror(errno), " (", errno, ")");
  }

    
  if (::close(fd) == -1) {
    AT_ERROR("Error closing file <", filename, ">");
  }
     
  c10::reportMemoryUsageToProfiler(base_ptr_, size_, c10::Device(c10::DeviceType::CPU));
}

void OffsettableMMap::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  if (base_ptr_ == nullptr) {
    return;
  }
  if (pinned_ && munlock(base_ptr_, size_)) {
    AT_ERROR("could not unlock the pages");
  }
  if (munmap(base_ptr_, size_)) {
    AT_ERROR("could not unmap the file");
  }
  
}

static void deleteOffsettableMMap(void* ptr) {
  delete static_cast<OffsettableMMap*>(ptr);
}

at::DataPtr OffsettableMMap::makeDataPtr(const char *filename,  size_t size, size_t offset, bool pin) {
  auto* allocatorOnHeap = new OffsettableMMap(filename, size, offset, pin);
  return {allocatorOnHeap->pointer(), allocatorOnHeap, &deleteOffsettableMMap, at::DeviceType::CPU};
}

// Offset mmap end

extern "C" {
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_cuda_1index(JNIEnv *env, jobject thisObj, jshort index) {
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
  JNIEXPORT jboolean JNICALL Java_aten_TensorOptions_isSparse(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    bool tpe = tensorOptions->is_sparse();
     return tpe;
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_getNumGPUs(JNIEnv *env, jobject thisObj) { try{
    
    jlong ret = at::detail::getCUDAHooks().getNumGPUs();
    return ret;
     } catch (exception& e) {
      return 0;
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_cudnnAvailable(JNIEnv *env, jobject thisObj) { try{
    
    bool ret = at::detail::getCUDAHooks().getNumGPUs() > 0 && at::detail::getCUDAHooks().hasCuDNN();
    return ret;
     } catch (exception& e) {
      return false;
    }
    return 0;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_manual_1seed(JNIEnv *env, jobject thisObj, jlong seed) { try{
    
    at::manual_seed(seed);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_manual_1seed_1cuda(JNIEnv *env, jobject thisObj, jlong seed, jint device) { try{
    
      auto cuda_gen = globalContext().defaultGenerator(Device(at::kCUDA, device));
      {
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        cuda_gen.set_current_seed(seed);
      }
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_manual_1seed_1cpu(JNIEnv *env, jobject thisObj, jlong seed) { try{
    
    auto gen = globalContext().defaultGenerator(DeviceType::CPU);
    {
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
  }
  JNIEXPORT jint JNICALL Java_aten_TensorOptions_deviceIndex(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    int32_t tpe = tensorOptions->device().index();
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
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_toHalf(JNIEnv *env, jobject thisObj) { try{
    
    jclass cls = tensorOptionsClass;
    TensorOptions* tensorOptions = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    
    TensorOptions* t2 = new TensorOptions(tensorOptions->dtype(kHalf));
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
  JNIEXPORT jobject JNICALL Java_aten_TensorOptions_dtypeHalf(JNIEnv *env, jobject thisObj) {
    try{
    TensorOptions* tensorOptions =new TensorOptions((ScalarType)5);
    
     return allocateTensorOptions(env,tensorOptions);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }

  JNIEXPORT jobject JNICALL Java_aten_Tensor_options(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    TensorOptions* opt = new TensorOptions(tensor->options());

    return allocateTensorOptions(env,opt);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jint JNICALL Java_aten_Tensor_dim(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor *tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->dim();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_defined(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->defined();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_isCuda(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->is_cuda();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_useCount(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->use_count();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_weakUseCount(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->weak_use_count();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_numel(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->numel();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_elementSize(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return tensor->element_size();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jbyte JNICALL Java_aten_Tensor_scalarType(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return static_cast<jbyte>(tensor->scalar_type());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
  }
  JNIEXPORT jstring JNICALL Java_aten_Tensor_nativeToString(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    return env->NewStringUTF(tensor->toString().c_str());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jstring JNICALL Java_aten_TensorOptions_nativeToString(JNIEnv *env, jobject thisObj) {
    try{
    TensorOptions* op = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    return env->NewStringUTF(c10::toString(*op).c_str());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_sizes(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    std::vector<int64_t> s = tensor->sizes().vec();
    return vecToJni(env,s);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT jlongArray JNICALL Java_aten_Tensor_strides(JNIEnv *env, jobject thisObj) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    std::vector<int64_t> s = tensor->strides().vec();
    return vecToJni(env,s);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_copyFrom(JNIEnv *env, jobject thisObj, jobject other, jboolean nonblocking) {
    try{
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor* otherTensor = reinterpret_cast<Tensor*>(env->GetLongField( other, tensorPointerFid));
    tensor->copy_(*otherTensor,nonblocking);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_print(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor->print();
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_mul_1(JNIEnv *env, jobject thisObj, jdouble d) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor->mul_(d);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_mul_1l_1(JNIEnv *env, jobject thisObj, jlong d) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor->mul_((int64_t)d);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelexpand_1as(JNIEnv *env, jobject thisObj, jobject other) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor1 = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor* tensor2 = reinterpret_cast<Tensor*>(env->GetLongField( other, tensorPointerFid));
    Tensor tensor3 = tensor1->expand_as(*tensor2);

    return  reinterpret_cast<jlong>(new Tensor(tensor3));
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelto_1dense(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor1 = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor tensor3 = tensor1->to_dense();

    return reinterpret_cast<jlong>(new Tensor(tensor3));
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelindices(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor1 = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor tensor3 = tensor1->indices();

    return  reinterpret_cast<jlong>(new Tensor(tensor3));
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelvalues(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor1 = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor tensor3 = tensor1->values();

    return reinterpret_cast<jlong>(new Tensor(tensor3));
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelcoalesce(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor1 = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor tensor3 = tensor1->coalesce();

    return  reinterpret_cast<jlong>(new Tensor(tensor3));
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelrepeat(JNIEnv *env, jobject thisObj, jlongArray repeat) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor1 = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    int64_t* longs = (int64_t*)env->GetLongArrayElements(repeat, nullptr);
    jsize length = env->GetArrayLength(repeat);
    IntArrayRef intarrayref = IntArrayRef(longs,length);
    Tensor tensor2 = tensor1->repeat(intarrayref);
    env->ReleaseLongArrayElements(repeat,(jlong*)longs,0);
    return  reinterpret_cast<jlong>(new Tensor(tensor2));
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_add_1(JNIEnv *env, jobject thisObj, jdouble other, jdouble alpha) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor->add_(other,alpha);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_add_1l_1(JNIEnv *env, jobject thisObj, jlong other, jlong alpha) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    tensor->add_((int64_t)other,(int64_t)alpha);
    return;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_TensorOptions_releaseNative(JNIEnv *env, jobject thisObj) {try{
    
    TensorOptions* op = reinterpret_cast<TensorOptions*>(env->GetLongField( thisObj, tensorOptionsPointerFid));
    delete op;    

    return ;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
  }
  JNIEXPORT void JNICALL Java_aten_Tensor_releaseNative(JNIEnv *env, jobject thisObj) {try{
    
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
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor->scalar_type()) != 6 || !tensor->is_contiguous() || !tensor->is_non_overlapping_and_dense() || tensor->data_ptr() == nullptr || len != tensor->numel() || tensor->is_cuda() || tensor->is_sparse()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      float* ptr = reinterpret_cast<float*>(tensor->data_ptr());
      for (int64_t i = 0;i < len;i++){
        ptr[i] = in[i];
      }
      env->ReleaseFloatArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromFloatArrayAtOffset(JNIEnv *env, jobject thisObj, jfloatArray datain, jlong offset) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor->scalar_type()) != 6 || !tensor->is_contiguous() || !tensor->is_non_overlapping_and_dense() || tensor->data_ptr() == nullptr || len+offset > tensor->numel() || tensor->is_cuda() || tensor->is_sparse()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      float* ptr = reinterpret_cast<float*>(tensor->data_ptr());
      for (int64_t i = 0;i < len;i++){
        ptr[i+offset] = in[i];
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
    Tensor tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid))->contiguous();
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 6 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel() || tensor.is_cuda() || tensor.is_sparse()) {
      return false;
    } else {
      float* in = env->GetFloatArrayElements(datain, nullptr);
      float* ptr = reinterpret_cast<float*>(tensor.data_ptr());
      for (int64_t i = 0;i < len;i++){
        in[i] =  ptr[i]  ;
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
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor->scalar_type()) != 4 || !tensor->is_contiguous() || !tensor->is_non_overlapping_and_dense() || tensor->data_ptr() == nullptr || len != tensor->numel() || tensor->is_cuda() || tensor->is_sparse()) {
      return false;
    } else {
      jlong* in = env->GetLongArrayElements(datain, nullptr);
      int64_t* ptr = reinterpret_cast<int64_t*>(tensor->data_ptr());
      for (int64_t i = 0;i < len;i++){
        ptr[i] = in[i];
      }
      env->ReleaseLongArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
 JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromLongArrayAtOffset(JNIEnv *env, jobject thisObj, jlongArray datain, jlong offset ) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor->scalar_type()) != 4 || !tensor->is_contiguous() || !tensor->is_non_overlapping_and_dense() || tensor->data_ptr() == nullptr || len+offset > tensor->numel() || tensor->is_cuda() || tensor->is_sparse()) {
      return false;
    } else {
      
      jlong* in = env->GetLongArrayElements(datain, nullptr);
      int64_t* ptr = reinterpret_cast<int64_t*>(tensor->data_ptr());
      for (int64_t i = 0;i < len;i++){
        ptr[i+offset] = in[i];
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
    Tensor tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid))->contiguous();
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 4 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel() || tensor.is_cuda() || tensor.is_sparse()) {
      return false;
    } else {
      jlong* in = env->GetLongArrayElements(datain, nullptr);
      int64_t* ptr = reinterpret_cast<int64_t*>(tensor.data_ptr());
      for (int64_t i = 0;i < len;i++){
        in[i] = ptr[i];
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
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor->scalar_type()) != 7 || !tensor->is_contiguous() || !tensor->is_non_overlapping_and_dense() || tensor->data_ptr() == nullptr || len != tensor->numel() || tensor->is_cuda() || tensor->is_sparse()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      double* ptr = reinterpret_cast<double*>(tensor->data_ptr());
      for (int64_t i = 0;i < len;i++){
        ptr[i] = in[i];
      }
      env->ReleaseDoubleArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jboolean JNICALL Java_aten_Tensor_copyFromDoubleArrayAtOffset(JNIEnv *env, jobject thisObj, jdoubleArray datain, jlong offset) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor->scalar_type()) != 7 || !tensor->is_contiguous() || !tensor->is_non_overlapping_and_dense() || tensor->data_ptr() == nullptr || len+offset > tensor->numel() || tensor->is_cuda() || tensor->is_sparse()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      double* ptr = reinterpret_cast<double*>(tensor->data_ptr());
      for (int64_t i = 0;i < len;i++){
        ptr[i+offset] = in[i];
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
    Tensor tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid))->contiguous();
    
      long len = env->GetArrayLength(datain);
    if (static_cast<int8_t>(tensor.scalar_type()) != 7 || !tensor.is_contiguous() || !tensor.is_non_overlapping_and_dense() || tensor.data_ptr() == nullptr || len != tensor.numel() || tensor.is_cuda() || tensor.is_sparse()) {
      return false;
    } else {
      double* in = env->GetDoubleArrayElements(datain, nullptr);
      double* ptr = reinterpret_cast<double*>(tensor.data_ptr());
      for (int64_t i = 0;i < len;i++){
        in[i] = ptr[i];
      }
      env->ReleaseDoubleArrayElements(datain,in,0);
      return true;
    }
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
  }
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelcpu(JNIEnv *env, jobject thisObj) {try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor retTensor = tensor->cpu();
    
      return allocateTensor(env,retTensor);
     } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }
  
  JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelcuda(JNIEnv *env, jobject thisObj) {
    try{
    
    jclass cls = tensorClass;
    Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
    Tensor retTensor = tensor->cuda();
    
    return allocateTensor(env,retTensor);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
  }

  JNIEXPORT void JNICALL Java_aten_Tensor_releaseAllNative(JNIEnv *env, jobject thisObj ,jobjectArray jniparam_tensors) {try{
  
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



JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelto(JNIEnv *env, jobject thisObj ,jobject jniparam_options, jboolean jniparam_non_blocking, jboolean jniparam_copy) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions* jniparam_options_c = reinterpret_cast<TensorOptions*>(jniparam_options_pointer);

   jclass tensor_cls = tensorClass;
   Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
      
  Tensor result =  tensor->to(*jniparam_options_c,jniparam_non_blocking,jniparam_copy);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  
    return ret_addressreturnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
}
JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelpin(JNIEnv *env, jobject thisObj ) {try{
  
   Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
      
  Tensor result =  tensor->pin_memory();
  
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
  
    return ret_addressreturnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
}
JNIEXPORT jboolean JNICALL Java_aten_Tensor_is_1pinned(JNIEnv *env, jobject thisObj ) {try{
  
   Tensor* tensor = reinterpret_cast<Tensor*>(env->GetLongField( thisObj, tensorPointerFid));
      
  jboolean result =  tensor->is_pinned();
  
    return result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return false;
}

JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelscalarDouble(JNIEnv *env, jobject thisObj ,jdouble jniparam_s,jobject jniparam_options) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions* jniparam_options_c = reinterpret_cast<TensorOptions*>(jniparam_options_pointer);
      

  Tensor result =  at::scalar_tensor(jniparam_s,*jniparam_options_c);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
    return ret_addressreturnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
}
JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelscalarLong(JNIEnv *env, jobject thisObj ,jlong jniparam_s,jobject jniparam_options) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions* jniparam_options_c = reinterpret_cast<TensorOptions*>(jniparam_options_pointer);
      
  int64_t p = (int64_t)jniparam_s;
  Tensor result =  at::scalar_tensor(p,*jniparam_options_c);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
    return ret_addressreturnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
}
JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelscalarFloat(JNIEnv *env, jobject thisObj ,jfloat jniparam_s,jobject jniparam_options) {try{
  

   jclass jniparam_options_class = tensorOptionsClass;
   jfieldID jniparam_options_fidNumber = tensorOptionsPointerFid;
   jlong jniparam_options_pointer = env->GetLongField( jniparam_options, jniparam_options_fidNumber);
   TensorOptions* jniparam_options_c = reinterpret_cast<TensorOptions*>(jniparam_options_pointer);
      

  Tensor result =  at::scalar_tensor(jniparam_s,*jniparam_options_c);
  

   
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
    return ret_addressreturnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return NULL;
}

JNIEXPORT void JNICALL Java_aten_Tensor_addmm_1out_1transposed2(JNIEnv *env, jobject thisObj ,jobject jniparam_out,jobject jniparam_self,jobject jniparam_mat1,jobject jniparam_mat2,jdouble jniparam_beta,jdouble jniparam_alpha) {try{
  
   jclass jniparam_out_class = tensorClass;
   jfieldID jniparam_out_fidNumber = tensorPointerFid;
   jlong jniparam_out_pointer = env->GetLongField( jniparam_out, jniparam_out_fidNumber);
   Tensor* jniparam_out_c = reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor* jniparam_self_c = reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor* jniparam_mat1_c = reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor* jniparam_mat2_c = reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::addmm_out(*jniparam_out_c,*jniparam_self_c,*jniparam_mat1_c,jniparam_mat2_c->t(),jniparam_beta,jniparam_alpha);
  
      






   
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
   Tensor* jniparam_out_c = reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor* jniparam_self_c = reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor* jniparam_mat1_c = reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor* jniparam_mat2_c = reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::addmm_out(*jniparam_out_c,*jniparam_self_c,jniparam_mat1_c->t(),*jniparam_mat2_c,jniparam_beta,jniparam_alpha);
  
      






   
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
   Tensor* jniparam_out_c = reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor* jniparam_self_c = reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor* jniparam_mat1_c = reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor* jniparam_mat2_c = reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::baddbmm_out(*jniparam_out_c,*jniparam_self_c,*jniparam_mat1_c,jniparam_mat2_c->transpose(1,2),jniparam_beta,jniparam_alpha);
  
      






   
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
   Tensor* jniparam_out_c = reinterpret_cast<Tensor*>(jniparam_out_pointer);
      

   jclass jniparam_self_class = tensorClass;
   jfieldID jniparam_self_fidNumber = tensorPointerFid;
   jlong jniparam_self_pointer = env->GetLongField( jniparam_self, jniparam_self_fidNumber);
   Tensor* jniparam_self_c = reinterpret_cast<Tensor*>(jniparam_self_pointer);
      

   jclass jniparam_mat1_class = tensorClass;
   jfieldID jniparam_mat1_fidNumber = tensorPointerFid;
   jlong jniparam_mat1_pointer = env->GetLongField( jniparam_mat1, jniparam_mat1_fidNumber);
   Tensor* jniparam_mat1_c = reinterpret_cast<Tensor*>(jniparam_mat1_pointer);
      

   jclass jniparam_mat2_class = tensorClass;
   jfieldID jniparam_mat2_fidNumber = tensorPointerFid;
   jlong jniparam_mat2_pointer = env->GetLongField( jniparam_mat2, jniparam_mat2_fidNumber);
   Tensor* jniparam_mat2_c = reinterpret_cast<Tensor*>(jniparam_mat2_pointer);
      



   at::baddbmm_out(*jniparam_out_c,*jniparam_self_c,jniparam_mat1_c->transpose(1,2),*jniparam_mat2_c,jniparam_beta,jniparam_alpha);
  
      






   
    return;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return ;
}

JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelones_1like(JNIEnv *env, jobject thisObj ,jlong jniparam_self) {try{
  
   
   Tensor* jniparam_self_c = reinterpret_cast<Tensor*>(jniparam_self);
      

  


  Tensor result =  at::ones_like(*jniparam_self_c);
  
      



   
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
 
   jlong returnable_result = ret_addressreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
}
JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelzeros_1like(JNIEnv *env, jobject thisObj ,jlong jniparam_self) {try{
  
   
   Tensor* jniparam_self_c = reinterpret_cast<Tensor*>(jniparam_self);
      

  


  Tensor result =  at::zeros_like(*jniparam_self_c);
  
      



   
  jclass ret_clsreturnable_result = tensorClass;
  Tensor* result_on_heapreturnable_result = new Tensor(result);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
 
   jlong returnable_result = ret_addressreturnable_result;
    return returnable_result;

    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
}

JNIEXPORT jlong JNICALL Java_aten_Tensor_lowlevelfrom_1file(JNIEnv *env, jobject thisObj ,jstring path, jlong offset, jlong len, jbyte scalarType, jboolean pin) {try{
  
  std::string pathAsStdString = jstring2string(env,path);

  auto dtype = TensorOptions((ScalarType)scalarType).dtype();

  int64_t numel = len / dtype.itemsize();
  if (len % dtype.itemsize() != 0) {
    throwRuntimeException(env,"Length (in bytes) is not a multiple of itemsize of dtype.");
  }
  auto storage_impl = c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      (int64_t) len,
      OffsettableMMap::makeDataPtr(pathAsStdString.c_str(), len,offset,pin),
      /*allocator=*/nullptr,
      /*resizable=*/false);

  auto tensor = at::detail::make_tensor<at::TensorImpl>(
      storage_impl, at::DispatchKey::CPU, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous({numel});

   
  Tensor* result_on_heapreturnable_result = new Tensor(tensor);
  jlong ret_addressreturnable_result = reinterpret_cast<jlong>(result_on_heapreturnable_result);
 
   return ret_addressreturnable_result;
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
}

JNIEXPORT void JNICALL Java_aten_Tensor_cudaCachingAllocatorSetMemoryFraction(JNIEnv *env, jobject thisObj ,jdouble fraction, jint device) {try{
  
      c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction,device);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    
}
JNIEXPORT void JNICALL Java_aten_Tensor_setPinnedMemoryAllocator(JNIEnv *env, jobject thisObj ) {try{
  
      c10::SetCPUAllocator(at::cuda::getPinnedMemoryAllocator());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    
}
JNIEXPORT void JNICALL Java_aten_Tensor_setDefaultAllocator(JNIEnv *env, jobject thisObj ) {try{
  
      c10::SetCPUAllocator(c10::GetDefaultCPUAllocator());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    
}

  // private static native void lowsetCurrentCUDAStream(long cudaStream);
JNIEXPORT void JNICALL Java_aten_CudaStream_lowlevelsynchronize(JNIEnv *env, jobject thisObj, jlong packed ) {try{
      
    c10::cuda::CUDAStream stream = c10::cuda::CUDAStream::unpack(reinterpret_signed_to_unsigned(packed));
    stream.synchronize();
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    
}
JNIEXPORT jlong JNICALL Java_aten_CudaStream_lowlevelgetStreamFromPool(JNIEnv *env, jobject thisObj, jboolean isHighPriority, jbyte device ) {try{
      
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(isHighPriority,device);
    return reinterpret_unsigned_to_signed(stream.pack());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
}
JNIEXPORT jlong JNICALL Java_aten_CudaStream_lowlevelgetDefaultCUDAStream(JNIEnv *env, jobject thisObj,  jbyte device ) {try{
      
    c10::cuda::CUDAStream stream = c10::cuda::getDefaultCUDAStream(device);
    return reinterpret_unsigned_to_signed(stream.pack());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return 0;
}
JNIEXPORT void JNICALL Java_aten_CudaStream_lowlevelsetCurrentCUDAStream(JNIEnv *env, jobject thisObj,  jlong packed ) {try{
      
    c10::cuda::CUDAStream stream = c10::cuda::CUDAStream::unpack(reinterpret_signed_to_unsigned(packed));
    c10::cuda::setCurrentCUDAStream(stream);
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
}
 JNIEXPORT jstring JNICALL Java_aten_CudaStream_nativeToString(JNIEnv *env, jobject thisObj, jlong packed) {
    try{
        c10::cuda::CUDAStream stream = c10::cuda::CUDAStream::unpack(reinterpret_signed_to_unsigned(packed));
        std::stringstream ss;

        ss<< stream;

    return env->NewStringUTF(ss.str().c_str());
    } catch (exception& e) {
      throwRuntimeException(env,e.what() );
    }
    return nullptr;
  }

}