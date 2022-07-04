#include <jni.h>   

extern jclass tensorClass;
extern jfieldID tensorPointerFid;
extern jclass tensorOptionsClass;
extern jmethodID tensorOptionsCtor;
extern jfieldID tensorOptionsPointerFid;
extern jclass longClass;
extern jmethodID longCtor;

extern jclass ncclCommClass;
extern jfieldID ncclCommPointerFid;
extern jmethodID ncclCommCtor;


std::string jstring2string(JNIEnv *env, jstring jStr);