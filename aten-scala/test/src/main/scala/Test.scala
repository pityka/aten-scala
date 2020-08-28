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
  assert(tensor1.weakUseCount() == 4)
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

  if (cuda) {
    tensor4.options.cuda_index(0)
  }

  val tensorLong = ATen.eye_1(2,2,TensorOptions.dtypeLong)
  val longA  =Array.ofDim[Long](4)
  val longA2  =Array.ofDim[Long](4)
  tensorLong.copyToLongArray(longA)
  assert(longA.toVector == Vector(1L,0L,0L,1L))
  longA(1) = 12
  tensorLong.copyFromLongArray(longA)
  tensorLong.copyToLongArray(longA2)
  assert(longA2.toVector == Vector(1L,12L,0L,1L))
  val argm = ATen.argmax(tensor5,0,false)
  assert(argm.sizes.toList == List(2))
  val array1 = Array.ofDim[Long](argm.sizes.apply(0).toInt)
  argm.copyToLongArray(array1)
  assert(array1.toVector == Vector(0L,1L))
  assert(argm.options.scalarTypeByte == 4L)


  {
    val tensorDouble = ATen.eye_1(2,2,TensorOptions.dtypeLong.toDouble)
  val tensorFloat = ATen.eye_1(2,2,TensorOptions.dtypeLong.toFloat)
  val t3 = tensorFloat.to(TensorOptions.dtypeDouble,true)
  println(t3.options)
  val t4 = tensorFloat.to(TensorOptions.dtypeDouble,false)
  println(t4.options)
  val tensorLong = ATen.eye_1(2,2,TensorOptions.dtypeLong.toFloat.toLong)
  val tensorLong2 = Tensor.scalarLong(1L, TensorOptions.dtypeLong)
  Tensor.releaseAll(Array(tensorDouble,tensorFloat, tensorLong2))
  }

  {
    val e = ATen.ones(Array(2,3),TensorOptions.dtypeDouble)
    val t = ATen.transpose(e,0,1)
    val a  =Array.ofDim[Double](6)
    t.copyToDoubleArray(a)
    assert(t.sizes.toList == List(3L,2L))
    println(a.toVector)
    assert(a.toVector == Vector(1d,1d,1d,1d,1d,1d))
    assert(t.options.isCPU)
    assert(!t.options.isCuda)
    assert(t.options.deviceIndex == 1)
  }
  
  {
    val e = ATen.ones(Array(2,3),TensorOptions.dtypeDouble)
    assert(e.scalarTypeByte == 7)
    val t = ATen.t(e)
    val a  =Array.ofDim[Double](6)
    t.copyToDoubleArray(a)
    assert(t.sizes.toList == List(3L,2L))
    println(a.toVector)
    assert(a.toVector == Vector(1d,1d,1d,1d,1d,1d))
    assert(t.options.isCPU)
    assert(!t.options.isCuda)
    assert(t.options.deviceIndex == 1)
  }

  println("CUDNN avail: "+Tensor.cudnnAvailable)
  // println("Has cuda: "+)
  println("done")

  Tensor.manual_seed(82L)
  println("Num gpus: "+Tensor.getNumGPUs)

  val expandto = ATen.ones(Array(3,3),TensorOptions.d)
  val expanded = ATen.ones(Array(1,1),TensorOptions.d).expand_as(expandto)
  assert(expanded.sizes.toList == List(3,3))

  {
    val e = ATen.ones(Array(2,3),TensorOptions.dtypeDouble)
    var i = 0 
    val N = 10000000
    while (i < N) {
      e.dim()
      i+=1 
    }
    i = 0
    val t1 = System.nanoTime()
    while (i < N) {
      e.dim()
      i+=1 
    }
    println("time per call dim: "+(System.nanoTime - t1)/(1E9*N))
  }

  {
    val e = ATen.ones(Array(2,3),TensorOptions.dtypeDouble)
    var i = 0 
    val N = 10000000
    while (i < N) {
      e.dim()
      i+=1 
    }
    i = 0
    val t1 = System.nanoTime()
    while (i < N) {
      e.sizes()
      i+=1 
    }
    println("time per call sizes: "+(System.nanoTime - t1)/(1E9*N))
  }

  {
    val topt = TensorOptions.dtypeDouble
    val d = Array(2L,3L)
    var i = 0 
    val N = 1000000
    while (i < N) {
      val e = ATen.ones(d,topt)
      i+=1 
    }
    i = 0
    val t1 = System.nanoTime()
    while (i < N) {
      val e = ATen.ones(d,topt)
      i+=1 
    }
    println("time per call ones: "+(System.nanoTime - t1)/(1E9*N))
  }

  {
    val topt = TensorOptions.dtypeDouble
    val d = Array(2L,3L)
    val e = ATen.ones(d,topt)
    var i = 0 
    val N = 1000000
    while (i < N) {
      ATen.zero_(e)
      i+=1 
    }
    i = 0
    val t1 = System.nanoTime()
    while (i < N) {
      ATen.zero_(e)
      i+=1 
    }
    println("time per call zero_: "+(System.nanoTime - t1)/(1E9*N))
  }

  {
    val topt = TensorOptions.dtypeDouble
    val d = Array(3000L,3000L)
    val t1 = ATen.ones(d,topt)
    val t2 = ATen.ones(d,topt)
    val v1 = t1.weakUseCount
    val c = ATen.cat(Array(t1,t2),0L)
    assert(v1 == t1.weakUseCount)
    t1.release 
    t2.release 
    c.release
    
  }



}
