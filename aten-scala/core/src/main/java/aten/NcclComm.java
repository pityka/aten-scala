package aten;

public class NcclComm {
	static {
		Load.load();
	}
	final long pointer;

	private NcclComm(long p) {
		pointer = p;
	}

	@Override
	public String toString() {
		return "NcclComm(@" + pointer + ")";
	}

	public static native byte[] get_unique_id();

	public native void comm_destroy();

	public static NcclComm comm_init_rank(int nranks, byte[] comm_id, int rank) {
		return new NcclComm(lowlevelcomm_init_rank(nranks, comm_id, rank));
	}

	private static native long lowlevelcomm_init_rank(int nranks, byte[] comm_id, int rank);

	public static void broadcast(Tensor[] tensors, NcclComm[] comms) {
		long[] ts = new long[tensors.length];
		for (int i = 0; i < ts.length; i++) {
			ts[i] = tensors[i].pointer;
		}

		long[] cs = new long[comms.length];
		for (int i = 0; i < cs.length; i++) {
			cs[i] = comms[i].pointer;
		}

		lowlevelbroadcast(ts, cs);

	}

	public static void reduce(Tensor[] inputs, Tensor output, int rootRank, int op, NcclComm[] comms) {
		long[] ts = new long[inputs.length];
		for (int i = 0; i < ts.length; i++) {
			ts[i] = inputs[i].pointer;
		}

		long[] cs = new long[comms.length];
		for (int i = 0; i < cs.length; i++) {
			cs[i] = comms[i].pointer;
		}

		lowlevelreduce(ts, output.pointer, rootRank, op, cs);

	}

	private static native void lowlevelbroadcast(long[] tensors, long[] comms);

	private static native void lowlevelreduce(long[] tensors, long output, int rootRank, int op, long[] comms);

}