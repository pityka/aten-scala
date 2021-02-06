package aten;

public class Tensor {
  static {
    Load.load();
  }
  final long pointer;

  private Tensor(long cPtr) {
    pointer = cPtr;
  }

  protected static Tensor factory(long cPtr) {
    Tensor t = new Tensor(cPtr);
    TensorTrace.recordAllocation(t);
    return t;
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

  private native long lowlevelcuda();
  public Tensor cuda() {
    return Tensor.factory(lowlevelcuda());
  }

  private native long lowlevelcpu();
  public Tensor cpu() {
    return Tensor.factory(lowlevelcpu());
  }

  public native void print();
  public native TensorOptions options();
  public static native boolean cudnnAvailable();
  public static native long getNumGPUs();
  public native byte scalarTypeByte();
  
  private native void releaseNative();
  public void release() {
    releaseNative();
    TensorTrace.recordRelease(this);
  }
  public static void releaseAll(Tensor[] tensors) {
    releaseAllNative(tensors);
    for (int x = 0; x < tensors.length; x++)  {
      TensorTrace.recordRelease(tensors[x]);
    }
  }
  private static native void releaseAllNative(Tensor[] tensors);
  
  private static native long lowlevelscalarDouble(double scalar, TensorOptions options);
  private static native long lowlevelscalarFloat(float scalar, TensorOptions options);
  private static native long lowlevelscalarLong(long scalar, TensorOptions options);
  public static Tensor scalarDouble(double scalar, TensorOptions options) {
    return Tensor.factory(lowlevelscalarDouble(scalar,options));
  }
  public static Tensor scalarFloat(float scalar, TensorOptions options) {
    return Tensor.factory(lowlevelscalarFloat(scalar,options));
  }
  public static  Tensor scalarLong(long scalar, TensorOptions options) {
    return Tensor.factory(lowlevelscalarLong(scalar,options));
  }

  private native long lowlevelto(TensorOptions op, boolean non_blocking, boolean copy);
  public Tensor to(TensorOptions op, boolean non_blocking, boolean copy) {
    return Tensor.factory(lowlevelto(op,non_blocking, copy));
  }

  public native boolean is_pinned();
  private native long lowlevelpin();
  public Tensor pin_memory() {
    return Tensor.factory(lowlevelpin());
  }

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
  public native boolean copyFromFloatArrayAtOffset( float[] data, long offset);
  public native boolean copyToFloatArray( float[] data);
 
  public native boolean copyFromDoubleArray( double[] data);
  public native boolean copyFromDoubleArrayAtOffset( double[] data, long offset);
  public native boolean copyToDoubleArray( double[] data);
 
  public native boolean copyFromLongArray( long[] data);
  public native boolean copyFromLongArrayAtOffset( long[] data, long offset);
  public native boolean copyToLongArray( long[] data);

  public native void mul_(double d);
  public native void add_(double other, double alpha);


  private native long lowlevelexpand_as(Tensor other);
  private native long lowlevelto_dense();
  private native long lowlevelvalues();
  private native long lowlevelindices();
  private native long lowlevelcoalesce();
  private native long lowlevelrepeat(long[] repeats);

 
  public Tensor expand_as(Tensor other) {
    return Tensor.factory(lowlevelexpand_as(other));
  }
  public Tensor to_dense() {
    return Tensor.factory(lowlevelto_dense());
  }
  public Tensor values() {
    return Tensor.factory(lowlevelvalues());
  }
  public Tensor indices() {
    return Tensor.factory(lowlevelindices());
  }
  public Tensor coalesce() {
    return Tensor.factory(lowlevelcoalesce());
  }
  public Tensor repeat(long[] repeats) {
    return Tensor.factory(lowlevelrepeat(repeats));
  }

  public static native void manual_seed(long seed);
  public static native void manual_seed_cpu(long seed);
  public static native void manual_seed_cuda(long seed, int device);

  public static native long lowlevelzeros_like(long tensor);
  public static Tensor zeros_like(Tensor tensor) {
      return Tensor.factory(lowlevelzeros_like(tensor.pointer));
  }
  public static native long lowlevelones_like(long tensor);
  public static Tensor ones_like(Tensor tensor) {
      return Tensor.factory(lowlevelones_like(tensor.pointer));
  }

}
