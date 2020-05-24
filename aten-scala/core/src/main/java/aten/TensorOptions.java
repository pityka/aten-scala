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
}