import aten._
import com.github.fommil.jni.JniLoader
import com.github.fommil.jni.JniNamer
object Test extends App {
val cuda = if (args.contains("--cuda")) true else false 

if (args.contains("--nccl")) {
  if (args.contains("root")) {
    val op = TensorOptions.f().cuda_index(0)
    val t = aten.ATen.zeros(Array(10),op)
    CudaStream.cudaSetDevice(0)
    val id = NcclComm.get_unique_id();
    println(java.util.Base64.getEncoder.encodeToString(id))
    val comm = NcclComm.comm_init_rank(2,id,0)
    println("block on broadcast, 0")
    NcclComm.broadcast(Array(t),Array(comm))
    println("block on reduce, 0")
    NcclComm.reduce(Array(t), t, 0, 0,Array(comm))
    val target1 = Array.ofDim[Float](10)
    val oncpu = t.cpu()
    assert(oncpu.copyToFloatArray(target1))
    assert(target1.toVector == Vector.fill(10)(2f))
  } else {
    val op = TensorOptions.f().cuda_index(0)
    val t = aten.ATen.ones(Array(10),op)
    CudaStream.cudaSetDevice(0)
    val id = java.util.Base64.getDecoder.decode(args.last)
    val comm = NcclComm.comm_init_rank(2,id,1)
    println("block on broadcast, 1")
    NcclComm.broadcast(Array(t),Array(comm))
    val target1 = Array.ofDim[Float](10)
    val oncpu = t.cpu()
    assert(oncpu.copyToFloatArray(target1))
    assert(target1.toVector == Vector.fill(10)(0f))
    val t2 = ATen.add_1(t,2d,1d)
    println("block on reduce, 1")
    NcclComm.reduce(Array(t2), t, 0, 0,Array(comm))
  }
} else {

if (cuda) {
  Tensor.setPinnedMemoryAllocator

  val stream = CudaStream.getStreamFromPool(false,0)
  println(stream)
  val default = CudaStream.getDefaultCUDAStream(0)
  CudaStream.setCurrentCUDAStream(stream)
  stream.synchronize()
  CudaStream.setCurrentCUDAStream(default)

  val id = NcclComm.get_unique_id();
  println("ncc id:")
  println(id.toVector)

}


  {
    val tmp = java.io.File.createTempFile("data","dat")
    val os = new java.io.FileOutputStream(tmp)
    os.write(Array.apply[Byte](1,1,-1,0,0,0,0,0,3,0,0,0,0,0,0,0))
    os.close 
    val tensor1 = Tensor.from_file(tmp.getAbsolutePath(), 0, 8, 1,true)
    val tensor1clone = ATen.clone(tensor1)
    tensor1clone.release
    val tensor1clone2 = ATen.clone(tensor1)
    println(tensor1.numel())
    val target1 = Array.ofDim[Byte](8)
    val target2 = Array.ofDim[Byte](8)
    assert(tensor1.copyToByteArray(target1))
    assert(target1.toVector == Vector(1,1,-1,0,0,0,0,0))
    target1(4) = -2
    tensor1clone2.copyFromByteArray(target1)
    assert(tensor1clone2.copyToByteArray(target2))
    assert(target1.toVector == Vector(1,1,-1,0,-2,0,0,0))
    assert(target2.toVector == Vector(1,1,-1,0,-2,0,0,0))

    tensor1.release 
    tensor1clone2.release
   

    
  }
  {
    val tmp = java.io.File.createTempFile("data","dat")
    val os = new java.io.FileOutputStream(tmp)
    os.write(Array.apply[Byte](1,1,0,0,0,0,0,0,3,0,0,0,0,0,0,0))
    os.close 
    val tensor1 = Tensor.from_file(tmp.getAbsolutePath(), 0, 8, 4,true)
    val tensor1clone = ATen.clone(tensor1)
    tensor1clone.release
    val target1 = Array.ofDim[Long](1)
    assert(tensor1.copyToLongArray(target1))
    assert(target1.toVector == Vector(257L))

    tensor1.release 
   

    
  }
  {
    val tmp = java.io.File.createTempFile("data","dat")
    val os = new java.io.FileOutputStream(tmp)
    os.write(Array.apply[Byte](1,1,0,0,
    0,0,0,0,
    1,0,0,0,
    0,0,0,0,
    1,1,1,1))
    os.close 
    
    scala.util.Try(Tensor.tensors_from_file(tmp.getAbsolutePath(), 0, 16, true,Array(4,4),Array(0,4),Array(8,8))).failed.get
    val tensors = Tensor.tensors_from_file(tmp.getAbsolutePath(), 0, 16, true,Array(4,4),Array(0,8),Array(8,8))
    val tensor1 = tensors(0)
    val tensor2 = tensors(1)
    assert(tensor1.numel == 1)
    assert(tensor2.numel == 1)
    assert(tensor1.sizes.toList == List(1))
    assert(tensor2.sizes.toList == List(1))
    val target1 = Array.ofDim[Long](1)
    assert(tensor1.copyToLongArray(target1))
    assert(target1.toVector == Vector(257))
    tensor1.release
    val target2 = Array.ofDim[Long](1)
    assert(tensor2.copyToLongArray(target2))
    assert(target2.toVector == Vector(1L))
    tensor2.release 
    
  }



  
  println("boo")

  val tensor1 = ATen.eye_0(3L, TensorOptions.dtypeFloat)
  println(tensor1)
  assert(tensor1.useCount() == 1)
  println("eye success")
  assert(TensorTrace.list.size == 0)
  TensorTrace.enable()
  val tensor2 = ATen._cast_Byte(tensor1, false)
  val tensor3 = ATen.eye_0(4L, TensorOptions.dtypeFloat)
  val currentLive = TensorTrace.list.toList.map(v => (v.getKey,v.getValue))
  assert(currentLive.map(_._1).contains(tensor2))
  assert(currentLive.map(_._1).contains(tensor3))
  assert(currentLive.size == 2)
  assert(currentLive.map(_._2).head.getShape != null)
  assert(tensor1.useCount() == 1)
  
  val opl = TensorOptions.l
  assert(opl.isLong)
  println(opl.release )
  // assert(opl.isFloat)

  println("cast success")
  val alignedArray = ATen.align_tensors(Array(tensor1, tensor1))
  assert(tensor1.useCount() == 3)

  assert(alignedArray.size == 2)
  assert(alignedArray.head.sizes.toList == List(3,3))
  println("align success")
  ATen.dropout_(tensor1, 1.0, false)
  println("dropout success")
  assert(tensor1.useCount() == 3)
  ATen.bartlett_window_1(1L, false, TensorOptions.dtypeFloat)
  println("bartless window successful")
  assert(tensor1.dim == 2)
  assert(tensor1.defined)
  val target1 = ATen.ones_like(tensor3,tensor3.options)
  target1.copyFrom(tensor3,true)
  assert(ATen.equal(target1,tensor3))

  tensor3.release
  Thread.sleep(100)
  assert(!TensorTrace.list.toList.map(v => (v.getKey)).contains(tensor3))

  assert(tensor1.useCount() == 3)
  assert(tensor1.weakUseCount() == 1)
  assert(tensor1.numel == 9)
  assert(!tensor1.isCuda)
  assert(tensor1.elementSize == 4)
  assert(tensor1.scalarType == 6) // 32bit float
  assert(tensor1.sizes.toVector == Seq(3, 3))
  assert(tensor1.strides.toVector == Seq(3, 1))
  assert(tensor1.useCount() == 3)
  val t1cpu = tensor1.cpu()
  assert(tensor1.useCount() == 4)
  t1cpu.release
  assert(tensor1.useCount() == 3)
  if (cuda){
  // this would move to cuda
    val t1c = tensor1.cuda()
    println(t1c)
    val target = Array.ofDim[Float](9)
    assert(tensor1.copyToFloatArray(target))
    println(target.toVector)
    assert(target(0) == 1)
    assert(target(4) == 1)
    assert(target(8) == 1)
  }
  tensor1.print
  val tensor4 = ATen.eye_1(4, 4, TensorOptions.dtypeFloat)

  ATen.bincount(ATen.zeros(Array(4), TensorOptions.dtypeLong),None,0)

  tensor4.print
  val op = tensor4.options
  println(op)

  assert(tensor1.useCount == 3)
  tensor1.release()
  

  val (eigA,eigB) = ATen.eig(tensor4, false)
  assert(eigA.sizes.toList == List(4,2))
  assert(eigB.sizes.toList == List(0))
  assert(tensor4.copyFromFloatArray(Array.ofDim[Float](16)))
  assert(!tensor4.copyFromFloatArray(Array.ofDim[Float](15)))
  val tensor5 = ATen.eye_1(2, 2, TensorOptions.d.cpu)
  val target = Array.ofDim[Double](4)
  tensor5.mul_(2d)
  tensor5.add_(1d,1d)
  ATen.add_1_l(tensor5,1L,1L)
  assert(tensor5.copyToDoubleArray(target))
  assert(target.toVector == Seq(3f, 1f, 1f, 3f))
  println(tensor4.options)

  val tensor4like = Tensor.zeros_like(tensor4)
  assert(ATen.equal(tensor4like,tensor4))
  val tensor4like1 = Tensor.ones_like(tensor4)

  if (cuda) {
    tensor4.options.cuda_index(0)
  }

  {
    val tensorLong = ATen.eye_1(2,2,TensorOptions.dtypeLong)
    val longA  =Array.ofDim[Long](2)
    assert(tensorLong.copyFromLongArrayAtOffset(longA,1L))
    assert(!tensorLong.copyFromLongArrayAtOffset(longA,3L))
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
  val t3 = tensorFloat.to(TensorOptions.dtypeDouble,true,true)
  if (!cuda) {
  assert(!t3.is_pinned)
  } else  {
    assert(t3.is_pinned)
  }
  println(t3.options)
  val t4 = tensorFloat.to(TensorOptions.dtypeDouble,false,false)
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
    assert(t.options.deviceIndex == -1)
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
    assert(!t.options.isSparse)
    assert(t.options.deviceIndex == -1)
  }

  {
    val values = ATen.ones(Array(2),TensorOptions.dtypeDouble)
    val indices = ATen.eye_0(2,TensorOptions.dtypeLong)
    val sparse = ATen.sparse_coo_tensor(indices,values,Array(4,4), TensorOptions.dtypeDouble).coalesce()
    assert(sparse.options.isSparse)
    assert(sparse.sizes.toList == List(4,4))
    
  }

  println("CUDNN avail: "+Tensor.cudnnAvailable)
  // println("Has cuda: "+)
  println("done")

  Tensor.manual_seed(82L)
  Tensor.manual_seed_cpu(83L)
  println("Num gpus: "+Tensor.getNumGPUs)

  val expandto = ATen.ones(Array(3,3),TensorOptions.d)
  val expanded = ATen.ones(Array(1,1),TensorOptions.d).expand_as(expandto)
  assert(expanded.sizes.toList == List(3,3))
  val repeatable= ATen.ones(Array(1,1),TensorOptions.d)
  val repeated = repeatable.repeat(Array(3,3))
  assert(repeated.sizes.toList == List(3,3))
  
  val repeated_jvm = Array.ofDim[Double](9)
  assert(repeated.copyToDoubleArray(repeated_jvm))
  val repeatable_jvm = Array.ofDim[Double](1)
  assert(repeatable.copyToDoubleArray(repeatable_jvm))
  assert(repeated_jvm.toVector.size==9)
  assert(repeatable_jvm.toVector.size==1)

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
}
