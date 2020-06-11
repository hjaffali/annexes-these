package mapleConnection;

public class SphereFunction implements ObjectiveFunctionReel {

	//SPHERE FONCTION NEGATIVE
	
	public SphereFunction() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Double apply(double[] t) {
		double sum = 0;
		for (int i=0; i<t.length; i++) {
			sum = sum - t[i]*t[i];
		}
		return sum;
	}

}
