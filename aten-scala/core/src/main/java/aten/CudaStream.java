package aten;

public class CudaStream {
  static {
    Load.load();
  }
  final long packed;

  private CudaStream(long p) {
    packed = p;
  }

  private native void lowlevelsynchronize(long pckd);
  private static native long lowlevelgetStreamFromPool(boolean isHighPriority, int device);
  private static native long lowgetDefaultCUDAStream(int device_index);
  private static native void lowsetCurrentCUDAStream(long cudaStream);

  public void synchronize() {
    lowlevelsynchronize(packed);
  }

  public static CudaStream getStreamFromPool(boolean isHighPriority, int device_index) {
    return new CudaStream(lowlevelgetStreamFromPool(isHighPriority,device_index));
  }

  public static CudaStream getDefaultCUDAStream(int device_index) {
    return new CudaStream(lowgetDefaultCUDAStream(device_index));
  }

  public static void setCurrentCUDAStream(CudaStream stream) {
    lowsetCurrentCUDAStream(stream.packed);
  }

}