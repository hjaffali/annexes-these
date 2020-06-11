package mapleConnection;

public class Det_3_qutrits implements ObjectiveFunctionReel {

	// DETERMINANT 3 QUTRITS SUR LA FORME NORMALE
	
	public Det_3_qutrits() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public Double apply(double[] y) {
		double[] t = new double[3];
		t[0] = y[0]/Math.sqrt(3);
		t[1] = y[1]/Math.sqrt(3);
		t[2] = y[2]/Math.sqrt(3);
		return Math.abs(-0.4e1 * Math.pow(t[0], 0.3e1) * Math.pow(t[1], 0.3e1) * Math.pow(t[2], 0.3e1) * Math.pow(t[0] + t[1] + t[2], 0.3e1) * Math.pow(Math.pow(t[0], 0.2e1) + Math.pow(t[1], 0.2e1) + Math.pow(t[2], 0.2e1) + 0.2e1 * t[0] * t[1] - t[0] * t[2] - t[1] * t[2], 0.3e1) * Math.pow(Math.pow(t[0], 0.2e1) + Math.pow(t[1], 0.2e1) + Math.pow(t[2], 0.2e1) - t[0] * t[1] + 0.2e1 * t[0] * t[2] - t[1] * t[2], 0.3e1) * Math.pow(Math.pow(t[0], 0.2e1) + Math.pow(t[1], 0.2e1) + Math.pow(t[2], 0.2e1) - t[0] * t[1] - t[0] * t[2] + 0.2e1 * t[1] * t[2], 0.3e1) * Math.pow(Math.pow(t[0], 0.2e1) + Math.pow(t[1], 0.2e1) + Math.pow(t[2], 0.2e1) - t[0] * t[1] - t[0] * t[2] - t[1] * t[2], 0.3e1));
	}

}
