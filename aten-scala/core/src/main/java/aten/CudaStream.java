package aten;

public class CudaStream {
  static {
    Load.load();
  }
  final long packed;

  private CudaStream(long p) {
    packed = p;
  }

  @Override
  public String toString() {
    return nativeToString(packed);
  }

  private native void lowlevelsynchronize(long pckd);
  private static native long lowlevelgetStreamFromPool(boolean isHighPriority, byte device_index);
  private static native long lowlevelgetDefaultCUDAStream(byte device_index);
  private static native void lowlevelsetCurrentCUDAStream(long cudaStream);
  private static native String nativeToString(long cudaStream);

  public void synchronize() {
    lowlevelsynchronize(packed);
  }

  public static CudaStream getStreamFromPool(boolean isHighPriority, byte device_index) {
    return new CudaStream(lowlevelgetStreamFromPool(isHighPriority,device_index));
  }

  public static CudaStream getDefaultCUDAStream(byte device_index) {
    return new CudaStream(lowlevelgetDefaultCUDAStream(device_index));
  }

  public static void setCurrentCUDAStream(CudaStream stream) {
    lowlevelsetCurrentCUDAStream(stream.packed);
  }

}