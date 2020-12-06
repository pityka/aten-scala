package aten;
public class TensorTraceData {
     final long birth = java.lang.System.nanoTime();
     long[] shape = null;
     boolean cpu = false;
     byte scalarType = 0;
     StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    public TensorTraceData(long[] shape1,  boolean cpu1, byte scalar1 ) {
      shape=shape1;
      cpu=cpu1;
      scalarType=scalar1;
    }
    public long getBirth() {
      return birth;
    }
    public boolean getCpu() {
      return cpu;
    }
    public byte getScalarType() {
      return scalarType;
    }
    public long[] getShape() {
      return shape;
    }
    public StackTraceElement[] getStackTrace() {
      return stackTrace;
    }
  }