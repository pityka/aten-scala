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

  public native TensorOptions cpu();
  public native TensorOptions cuda();
  public native TensorOptions cuda_index(int device);

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
}