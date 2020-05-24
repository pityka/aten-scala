package aten;

public class Tensor {
  static {
    Load.load();
  }
  private transient long pointer;

  protected Tensor(long cPtr) {
    pointer = cPtr;
  }

  protected Tensor() {
    pointer = 0;
  }

  protected static long getCPtr(Tensor obj) {
    return (obj == null) ? 0 : obj.pointer;
  }

  public native long dim();
  public native boolean defined();
  public native long useCount();
  public native long weakUseCount();
  public native String nativeToString();
  public native long[] sizes();
  public native long[] strides();
  public native long numel();
  public native long elementSize();
  public native byte scalarType();
  public native boolean isCuda();
  public native Tensor cuda();
  public native Tensor cpu();
  public native void print();
  
  public native void release();

  @Override
  public String toString() {
    return "Tensor at "+pointer+"; "+nativeToString();
  }
  
  public native void setToTensor(Tensor other);

  public native boolean copyFromFloatArray( float[] data);
  public native boolean copyToFloatArray( float[] data);
 
  public native boolean copyFromDoubleArray( double[] data);
  public native boolean copyToDoubleArray( double[] data);

}
