package aten;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.Set;

public class TensorTrace {

  private static ConcurrentHashMap<Tensor,TensorTraceData> allocated = new ConcurrentHashMap<Tensor,TensorTraceData>();
  
  private static boolean enabled = false;

  public static void enable() {
    enabled=true;
  }
  public static void disable() {
    enabled=false;
  }

  public static Map.Entry<Tensor,TensorTraceData>[] list() {
    return allocated.entrySet().toArray(new Map.Entry[0]);
  }

  // Only ever call this within the ctor of Tensor
  public static void recordAllocation(Tensor tensor) {
    if (!enabled) {
      return;
    } else {
      if (tensor.defined()) {
        TensorTraceData data = new TensorTraceData(tensor.sizes().clone(), !tensor.isCuda(), tensor.scalarType() );
        allocated.put(tensor,data);
      }
    }

  }

  public static void recordRelease(Tensor tensor) {
     if (!enabled) {
      return;
    } else {
      allocated.remove(tensor);
    }

  }
}