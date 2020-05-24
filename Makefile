prepare:
	cd parser && sbt bloopInstall;
	cd aten-scala && sbt bloopInstall;

wrapper.cpp: 
	cd parser; bloop run parser -- ../parseable.h ../wrapper.cpp ../aten-scala/core/src/main/java/aten/ATen.java


aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib: wrapper.cpp wrapper_manual.cpp
	 clang++ -std=c++14 -I $(JAVA_HOME)/include/ -I $(JAVA_HOME)/include/darwin/ -I libtorch_mac/include/ -lc10 -ltorch_global_deps -ltorch -ltorch_cpu -shared -undefined dynamic_lookup -o aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib wrapper_manual.cpp wrapper.cpp  libtorch_mac/lib/*a

test: aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib
		cd aten-scala; bloop run test 

publishLocal: aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib
		cd aten-scala && sbt publishLocal

all: test

clean:
	rm -rf wrapper.cpp;
	rm -rf aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib