package mapleConnection;

public class Discrim_4_qubits implements ObjectiveFunctionReel {

	//DISRIMINANT 4 QUBITS
	
	public Discrim_4_qubits() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Double apply(double[] t) {
		return Math.abs(-12288*t[1]*t[0]*t[0]*t[4]*t[4]*t[3]-1536*t[4]*t[0]*t[3]*t[3]*t[1]*t[1]-1536*t[1]*t[1]*Math.sqrt(6)*t[2]*t[2]*t[2]*t[4]+1536*t[1]*t[1]*t[2]*t[2]*t[3]*t[3]+9216*t[4]*t[4]*t[1]*t[1]*t[0]*Math.sqrt(6)*t[2]-12288*t[0]*t[0]*t[4]*t[4]*t[2]*t[2]+4608*t[1]*t[3]*t[3]*t[3]*t[0]*Math.sqrt(6)*t[2]+4608*t[4]*Math.sqrt(6)*t[2]*t[1]*t[1]*t[1]*t[3]+9216*t[2]*t[2]*t[2]*t[2]*t[0]*t[4]-1536*Math.sqrt(6)*t[2]*t[2]*t[2]*t[0]*t[3]*t[3]-4096*t[1]*t[1]*t[1]*t[3]*t[3]*t[3]-6912*t[4]*t[4]*t[1]*t[1]*t[1]*t[1]+4096*t[0]*t[0]*t[0]*t[4]*t[4]*t[4]-6912*t[0]*t[0]*t[3]*t[3]*t[3]*t[3]+9216*t[4]*t[0]*t[0]*t[3]*t[3]*Math.sqrt(6)*t[2]-30720*t[4]*t[2]*t[2]*t[1]*t[3]*t[0]);
	}
	
}