package aten;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.Set;

public class TensorOptionsTrace {

  private static ConcurrentHashMap<TensorOptions,TensorTraceData> allocated = new ConcurrentHashMap<TensorOptions,TensorTraceData>();
  
  private static boolean enabled = false;

  public static void enable() {
    enabled=true;
  }
  public static void disable() {
    enabled=false;
  }

  public static Map.Entry<TensorOptions,TensorTraceData>[] list() {
    return allocated.entrySet().toArray(new Map.Entry[0]);
  }

  public static void recordAllocation(TensorOptions tensor) {
    if (!enabled) {
      return;
    } else {
      long[] l = {};
      TensorTraceData data = new TensorTraceData(l, !tensor.isCuda(), tensor.scalarTypeByte() );
      allocated.put(tensor,data);
    }

  }

  public static void recordRelease(TensorOptions tensor) {
     if (!enabled) {
      return;
    } else {
      allocated.remove(tensor);
    }

  }
}