#include <jni.h>   

extern jclass tensorClass;
extern jfieldID tensorPointerFid;
extern jclass tensorOptionsClass;
extern jmethodID tensorOptionsCtor;
extern jfieldID tensorOptionsPointerFid;
extern jclass longClass;
extern jmethodID longCtor;

std::string jstring2string(JNIEnv *env, jstring jStr);