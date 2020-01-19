package org.ml4j.nn.dl4j.activationfunctions;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DefaultReluActivationFunctionImpl;

public class ActivationML4JReLU extends BaseML4JActivationFunction {

	/**
	 * Defualt serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public ActivationML4JReLU(MatrixFactory matrixFactory) {
		// TODO Could default to Nd4jMatrixFactory
		super(matrixFactory, new DefaultReluActivationFunctionImpl());
	}
}
