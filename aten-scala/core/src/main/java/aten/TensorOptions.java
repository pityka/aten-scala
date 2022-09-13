package aten;

public class TensorOptions {
  static {
    Load.load();
  }
  final transient long pointer;

  private TensorOptions(long cPtr) {
    pointer = cPtr;
    TensorOptionsTrace.recordAllocation(this);
  }

  private native void releaseNative();
  public void release() {
    releaseNative();
    TensorOptionsTrace.recordRelease(this);
  }

  protected static long getCPtr(TensorOptions obj) {
    return (obj == null) ? 0 : obj.pointer;
  }

  public static native TensorOptions dtypeFloat();
  public static native TensorOptions dtypeDouble();
  public static native TensorOptions dtypeLong();
  public static native TensorOptions dtypeByte();
  public static native TensorOptions dtypeHalf();

  public static TensorOptions d() {
    return TensorOptions.dtypeDouble();
  }
  public static TensorOptions f() {
    return TensorOptions.dtypeFloat();
  }
  public static TensorOptions l() {
    return TensorOptions.dtypeLong();
  }
  public static TensorOptions h() {
    return TensorOptions.dtypeHalf();
  }
  public static TensorOptions b() {
    return TensorOptions.dtypeByte();
  }
  public native TensorOptions toDouble();
  public native TensorOptions toLong();
  public native TensorOptions toFloat();
  public native TensorOptions toHalf();
  public native TensorOptions toByte();

  public native TensorOptions cpu();
  public native boolean isCPU();
  public native boolean isMps();
  public native boolean isCuda();
  public native boolean isSparse();
  public native TensorOptions cuda();
  public native TensorOptions cuda_index(short device);
  public native TensorOptions device(byte deviceType, byte deviceIndex);
  public native int deviceIndex();
  public native int deviceType();
  public native byte scalarTypeByte();
  public native String nativeToString();

  public String toString() {
    return "TensorOptions at "+pointer+"; "+nativeToString();
  }

  public static TensorOptions fromScalarType(byte i) {
    if (i == 6) {
      return TensorOptions.dtypeFloat();
    } else if (i == 7) {
      return TensorOptions.dtypeDouble();
    } else if (i == 5) {
      return TensorOptions.dtypeHalf();
    } else if (i == 4) {
      return TensorOptions.dtypeLong();
    } else if (i == 1) {
      return TensorOptions.dtypeByte();
    } else {
      throw new RuntimeException("unknown scalar type "+i);
    }
  }

  public boolean isHalf() {
    return scalarTypeByte() == 5;
  }
  public boolean isFloat() {
    return scalarTypeByte() == 6;
  }
  public boolean isDouble() {
    return scalarTypeByte() == 7;
  }
  public boolean isLong() {
    return scalarTypeByte() == 4;
  }
  public boolean isByte() {
    return scalarTypeByte() == 1;
  }

}