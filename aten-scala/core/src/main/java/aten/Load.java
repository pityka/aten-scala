package aten;

import com.github.fommil.jni.JniLoader;

public class Load {
  volatile static boolean loaded = false;

  public static synchronized void load() {
    if (!loaded) {
      String os = System.getProperty("os.name", "").toLowerCase();
      if (os.startsWith("linux")) {
        JniLoader.load("libatenscalajni.so");
      } else if  (os.startsWith("mac os x") || os.startsWith("darwin")) {
        JniLoader.load("libatenscalajni.dylib");
      } else {
        throw new RuntimeException("Operating system not supported or not recognized: "+os);
      }
      loaded = true;
      return;
    }
  }
}