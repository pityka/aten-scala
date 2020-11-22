package aten;

public class Tensor {
  static {
    Load.load();
  }
  final transient long pointer;

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
  public native TensorOptions options();
  public static native boolean cudnnAvailable();
  public static native long getNumGPUs();
  public native byte scalarTypeByte();
  
  public native void release();
  public static native void releaseAll(Tensor[] tensors);
  
  public static native Tensor scalarDouble(double scalar, TensorOptions options);
  public static native Tensor scalarFloat(float scalar, TensorOptions options);
  public static native Tensor scalarLong(long scalar, TensorOptions options);

  public native Tensor to(TensorOptions op, boolean copy);

  public static native void addmm_out_transposed1(Tensor out,Tensor self,Tensor mat1,Tensor mat2,double beta,double alpha);
  /* Same as Aten.addmm_out but mat2 will be transposed before addmm */
  public static native void addmm_out_transposed2(Tensor out,Tensor self,Tensor mat1,Tensor mat2,double beta,double alpha);
 
  public static native void baddbmm_out_transposed1(Tensor out,Tensor self,Tensor mat1,Tensor mat2,double beta,double alpha);
  public static native void baddbmm_out_transposed2(Tensor out,Tensor self,Tensor mat1,Tensor mat2,double beta,double alpha);

  @Override
  public String toString() {
    return "Tensor at "+pointer+"; "+nativeToString();
  }
  
  public native void copyFrom(Tensor other);

  public native boolean copyFromFloatArray( float[] data);
  public native boolean copyToFloatArray( float[] data);
 
  public native boolean copyFromDoubleArray( double[] data);
  public native boolean copyToDoubleArray( double[] data);
 
  public native boolean copyFromLongArray( long[] data);
  public native boolean copyToLongArray( long[] data);

  public native void mul_(double d);
  public native void add_(double other, double alpha);
  public native Tensor expand_as(Tensor other);
  public native Tensor to_dense();
  public native Tensor values();
  public native Tensor indices();
  public native Tensor coalesce();
  public native Tensor repeat(long[] repeats);

  public static native void manual_seed(long seed);

}
