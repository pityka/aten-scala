import aten._
import com.github.fommil.jni.JniLoader
import com.github.fommil.jni.JniNamer
object Test extends App {

  val cuda = if (args.contains("--cuda")) true else false 
  println("boo")

  val tensor1 = ATen.eye_0(3L, TensorOptions.dtypeFloat)
  println(tensor1)
  println("eye success")
  val tensor2 = ATen._cast_Byte(tensor1, false)
  val tensor3 = ATen.eye_0(4L, TensorOptions.dtypeFloat)
  println("cast success")
  ATen.align_tensors(Array(tensor1, tensor1))
  println("align success")
  ATen.dropout_(tensor1, 1.0, false)
  println("dropout success")
  ATen.bartlett_window_1(1L, false, TensorOptions.dtypeFloat)
  println("bartless window successful")
  assert(tensor1.dim == 2)
  assert(tensor1.defined)
  tensor1.setToTensor(tensor3)
  assert(tensor1.useCount() == 1)
  assert(tensor1.weakUseCount() == 6)
  println(tensor1.toString)
  assert(tensor1.numel == 9)
  assert(!tensor1.isCuda)
  assert(tensor1.elementSize == 4)
  assert(tensor1.scalarType == 6) // 32bit float
  assert(tensor1.sizes.deep.toVector == Seq(3, 3))
  assert(tensor1.strides.deep.toVector == Seq(3, 1))
  tensor1.cpu()
  if (cuda){
  // this would move to cuda
    val t1c = tensor1.cuda()
    println(t1c)
    val target = Array.ofDim[Float](9)
    assert(tensor1.copyToFloatArray(target))
    println(target.deep)
    assert(target(0) == 1)
    assert(target(4) == 1)
    assert(target(8) == 1)
  }
  tensor1.print
  val tensor4 = ATen.eye_1(4, 4, TensorOptions.dtypeFloat)
  tensor4.print

  assert(tensor1.useCount == 1)
  tensor1.release()

  println(ATen.eig(tensor4, false));
  assert(tensor4.copyFromFloatArray(Array.ofDim[Float](16)))
  assert(!tensor4.copyFromFloatArray(Array.ofDim[Float](15)))
  val tensor5 = ATen.eye_1(2, 2, TensorOptions.d.cpu)
  val target = Array.ofDim[Double](4)
  tensor5.mul_(2d)
  tensor5.add_(1d,1d)
  assert(tensor5.copyToDoubleArray(target))
  assert(target.deep.toSeq == Seq(3f, 1f, 1f, 3f))
  println(tensor4.options)

  val tensorLong = ATen.eye_1(2,2,TensorOptions.dtypeLong)
  val longA  =Array.ofDim[Long](4)
  val longA2  =Array.ofDim[Long](4)
  tensorLong.copyToLongArray(longA)
  assert(longA.toVector == Vector(1L,0L,0L,1L))
  longA(1) = 12
  tensorLong.copyFromLongArray(longA)
  tensorLong.copyToLongArray(longA2)
  assert(longA2.toVector == Vector(1L,12L,0L,1L))
  val argm = ATen.argmax(tensor5,1,false)
  assert(argm.sizes.toList == List(2))


  {
    val tensorDouble = ATen.eye_1(2,2,TensorOptions.dtypeLong.toDouble)
  val tensorFloat = ATen.eye_1(2,2,TensorOptions.dtypeLong.toFloat)
  val tensorLong = ATen.eye_1(2,2,TensorOptions.dtypeLong.toFloat.toLong)
  }

}
