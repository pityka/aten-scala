package aten;

public class TensorOptions {
  static {
    Load.load();
  }
  private transient long pointer;

  protected TensorOptions(long cPtr) {
    pointer = cPtr;
  }

  protected TensorOptions() {
    pointer = 0;
  }

  protected static long getCPtr(TensorOptions obj) {
    return (obj == null) ? 0 : obj.pointer;
  }

  public static native TensorOptions dtypeFloat();
  public static native TensorOptions dtypeDouble();
  public static native TensorOptions dtypeLong();

  public static TensorOptions d() {
    return TensorOptions.dtypeDouble();
  }
  public static TensorOptions f() {
    return TensorOptions.dtypeFloat();
  }
  public static TensorOptions l() {
    return TensorOptions.dtypeLong();
  }
  public native TensorOptions toDouble();
  public native TensorOptions toLong();
  public native TensorOptions toFloat();

  public native TensorOptions cpu();
  public native boolean isCPU();
  public native boolean isCuda();
  public native TensorOptions cuda();
  public native TensorOptions cuda_index(int device);
  public native short deviceIndex();
  public native byte scalarTypeByte();

  public static TensorOptions fromScalarType(byte i) {
    if (i == 6) {
      return TensorOptions.dtypeFloat();
    } else if (i == 7) {
      return TensorOptions.dtypeDouble();
    } else if (i == 4) {
      return TensorOptions.dtypeLong();
    } else {
      throw new RuntimeException("unknown scalar type "+i);
    }
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

}