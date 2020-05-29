docker-prepare:
	cd docker-build && docker build -t aten-scala-linux-build .
	cd docker-runtime && docker build -t aten-scala-linux-runtime .

prepare:
	cd parser && sbt bloopInstall;
	cd aten-scala && sbt bloopInstall;

wrapper.cpp: 
	cd parser; bloop run parser -- ../parseable.h ../wrapper.cpp ../aten-scala/core/src/main/java/aten/ATen.java

aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib: wrapper.cpp wrapper_manual.cpp
	 clang++ -std=c++14 -I $(JAVA_HOME)/include/ -I $(JAVA_HOME)/include/darwin/ -I libtorch_mac/include/ -lc10 -ltorch_global_deps -ltorch -ltorch_cpu -shared -undefined dynamic_lookup -o aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib wrapper_manual.cpp wrapper.cpp

aten-scala/jni-linux/src/main/resources/libatenscalajni.so: wrapper.cpp wrapper_manual.cpp
	 docker run -v `pwd`:/build aten-scala-linux-build /bin/bash -c "cd /build;  clang++ -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -I /usr/lib/jvm/java-8-openjdk-amd64/include/ -I /usr/lib/jvm/java-8-openjdk-amd64/include/linux/ -I libtorch_linux/include/ -L /usr/local/lib/python3.8/dist-packages/torch/lib/ -lc10 -ltorch_global_deps -ltorch -ltorch_cpu -ltorch_cuda -fPIC -shared -o aten-scala/jni-linux/src/main/resources/libatenscalajni.so wrapper_manual.cpp wrapper.cpp "

test: aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib
		cd aten-scala; bloop run test 

test-linux: aten-scala/jni-linux/src/main/resources/libatenscalajni.so
		docker run -v `pwd`:/build aten-scala-linux-runtime /bin/bash -c "cd /build/aten-scala; sbt 'test/run'"


publishLocal: aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib aten-scala/jni-linux/src/main/resources/libatenscalajni.so
		cd aten-scala && sbt publishLocal

all: test

clean:
	rm -rf wrapper.cpp;
	rm -rf aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib