package aten;

public class CudaStream {
  static {
    Load.load();
  }
  final long packedStreamId;
  final byte deviceIndex;

  private CudaStream(long p, byte deviceIndex_) {
    packedStreamId = p;
    deviceIndex = deviceIndex_;
  }

  @Override
  public String toString() {
    return nativeToString(packedStreamId, deviceIndex);
  }

  private native void lowlevelsynchronize(long pckd, byte deviceIndex);
  private static native long lowlevelgetStreamFromPool(boolean isHighPriority, byte device_index);
  private static native long lowlevelgetDefaultCUDAStream(byte device_index);
  private static native long lowlevelgetCurrentCUDAStream(byte device_index);
  private static native void lowlevelsetCurrentCUDAStream(long cudaStream, byte device_index);
  private static native String nativeToString(long cudaStream, byte deviceIndex);
  public static native void cudaSetDevice(int device);
  public static native int cudaGetDevice();

  public void synchronize() {
    lowlevelsynchronize(packedStreamId, deviceIndex);
  }

  public static CudaStream getStreamFromPool(boolean isHighPriority, byte device_index) {
    return new CudaStream(lowlevelgetStreamFromPool(isHighPriority,device_index), device_index);
  }

  public static CudaStream getDefaultCUDAStream(byte device_index) {
    return new CudaStream(lowlevelgetDefaultCUDAStream(device_index), device_index);
  }
  public static CudaStream getCurrentCUDAStream(byte device_index) {
    return new CudaStream(lowlevelgetCurrentCUDAStream(device_index), device_index);
  }

  public static void setCurrentCUDAStream(CudaStream stream) {
    lowlevelsetCurrentCUDAStream(stream.packedStreamId, stream.deviceIndex);
  }

}