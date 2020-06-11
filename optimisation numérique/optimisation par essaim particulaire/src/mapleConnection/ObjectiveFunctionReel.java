package mapleConnection;

import java.util.function.Function;

public interface ObjectiveFunctionReel extends Function<double[], Double> {
	
	public Double apply(double[] t);
	
}
