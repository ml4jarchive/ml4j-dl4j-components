/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.dl4j.activationfunctions;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DefaultDifferentiableActivationFunctionActivationImpl;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.activationfunctions.DefaultDifferentiableActivationFunctionComponentActivationImpl;
import org.ml4j.nn.components.activationfunctions.DefaultDifferentiableActivationFunctionComponentImpl;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentAdapter;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons1D;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContextImpl;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class BaseML4JActivationFunction extends BaseActivationFunction implements IActivation {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private DifferentiableActivationFunction ml4jActivationFunction;
	private MatrixFactory matrixFactory;
	
	public BaseML4JActivationFunction(MatrixFactory matrixFactory, DifferentiableActivationFunction ml4jActivationFunction) {
		this.ml4jActivationFunction = ml4jActivationFunction;
		// TODO Could default to Nd4jMatrixFactory
		this.matrixFactory = matrixFactory;
	}
	
	@Override
	public Pair<INDArray, INDArray> backprop(INDArray in, INDArray eps) {

		NeuronsActivation inputNeuronsActivation = fromNDArray(in);
		NeuronsActivation gradientNeuronsActivation = fromNDArray(eps);
		
		Neurons neurons = inputNeuronsActivation.getNeurons();
		
		DifferentiableActivationFunctionComponentAdapter ml4jActivationFunctionComponent
		 = new DefaultDifferentiableActivationFunctionComponentImpl(neurons, ml4jActivationFunction);
		
		
		// No output available, or needed, so set null.
		DifferentiableActivationFunctionActivation activationFunctionActivation
		 = new DefaultDifferentiableActivationFunctionActivationImpl(ml4jActivationFunction, inputNeuronsActivation, null);
		
		DifferentiableActivationFunctionComponentActivation activationFunctionComponentActivation
		 = new DefaultDifferentiableActivationFunctionComponentActivationImpl(ml4jActivationFunctionComponent, 
				 activationFunctionActivation, new NeuronsActivationContextImpl(matrixFactory, true));
		
		DirectedComponentGradient<NeuronsActivation> inputGradient = new DirectedComponentGradientImpl<>(gradientNeuronsActivation);
		
		DirectedComponentGradient<NeuronsActivation> outputGradient = activationFunctionComponentActivation.backPropagate(inputGradient);
				
		return new Pair<>(asNDArray(matrixFactory, outputGradient.getOutput()), null);
	}

	@Override
	public INDArray getActivation(INDArray input, boolean training) {
		
		DifferentiableActivationFunctionActivation activation = ml4jActivationFunction.activate(fromNDArray(matrixFactory, input, 
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, null, false), 
				new NeuronsActivationContextImpl(matrixFactory, training));
		
		return asNDArray(matrixFactory, activation.getOutput());
	}

	private NeuronsActivation fromNDArray(INDArray ndArray) {
		int rows = ndArray.rows();
		Neurons neurons = new Neurons1D(rows, false);
		return fromNDArray(matrixFactory, ndArray, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, neurons, false);
	}
	
	private NeuronsActivation fromNDArray(MatrixFactory matrixFactory, INDArray ndArray, NeuronsActivationFeatureOrientation featureOrientation, Neurons neurons, boolean imageActivation) {
		float[] rowsByRowsArray = ndArray.data().getFloatsAt(ndArray.offset(), ndArray.length());
		NeuronsActivation neuronsActivation = new NeuronsActivationImpl(neurons, matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray),
				featureOrientation);
		if (imageActivation) {
			return neuronsActivation.asImageNeuronsActivation((Neurons3D)neurons);
		} else {
			return neuronsActivation;
		}
	}
	
	private INDArray asNDArray(MatrixFactory matrixFactory, NeuronsActivation neuronsActivation) {
		Matrix matrix = neuronsActivation.getActivations(matrixFactory);
		return Nd4j.create(matrix.getRowByRowArray(), new int[] { matrix.getRows(), matrix.getColumns() });
	}


}
