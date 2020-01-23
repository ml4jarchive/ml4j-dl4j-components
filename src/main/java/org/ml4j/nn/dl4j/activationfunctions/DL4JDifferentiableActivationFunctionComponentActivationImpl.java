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

import java.util.Arrays;
import java.util.List;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Adapter to wrap a DL4J IActivation instance and the activation output, so
 * that they conform to a ML4J
 * DifferentiableActivationFunctionComponentActivation interface.
 * 
 * Allows DL4J-specific activation functions to be used with a ML4J component
 * graph or network.
 * 
 * @author Michael Lavelle
 *
 */
public class DL4JDifferentiableActivationFunctionComponentActivationImpl
		implements DifferentiableActivationFunctionComponentActivation {

	private NeuronsActivation inputActivation;
	private INDArray inputNDArray;
	private NeuronsActivation outputActivation;
	private IActivation dl4jActivationFunction;
	private MatrixFactory matrixFactory;
	private ActivationFunctionType activationFunctionType;
	NeuronsActivationFeatureOrientation dl4jFeatureOrientation;

	public DL4JDifferentiableActivationFunctionComponentActivationImpl(MatrixFactory matrixFactory,
			IActivation dl4jActivationFunction, ActivationFunctionType activationFunctionType,
			NeuronsActivation inputActivation, INDArray inputNDArray, NeuronsActivation outputActivation,
			NeuronsActivationFeatureOrientation dl4jFeatureOrientation) {
		this.inputActivation = inputActivation;
		this.inputNDArray = inputNDArray;
		this.outputActivation = outputActivation;
		this.dl4jActivationFunction = dl4jActivationFunction;
		this.activationFunctionType = activationFunctionType;
		this.matrixFactory = matrixFactory;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {

		INDArray gradientActivations = DL4JUtil.asNDArray(matrixFactory, gradient.getOutput(), dl4jFeatureOrientation);

		Pair<INDArray, INDArray> backProp = dl4jActivationFunction.backprop(inputNDArray, gradientActivations);
		if (backProp.getSecond() != null) {
			throw new IllegalStateException("Activation gradient for activation functions with weights not supported");
		}
		INDArray backPropFirst = backProp.getFirst();

		NeuronsActivation outputGradient = DL4JUtil.fromNDArray(matrixFactory, backPropFirst, dl4jFeatureOrientation,
				inputActivation.getFormat(), inputActivation.getNeurons());
		return new DirectedComponentGradientImpl<>(gradient.getTotalTrainableAxonsGradients(), outputGradient);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient da) {
		return da.backPropagateThroughFinalActivationFunction(activationFunctionType);
	}

	@Override
	public List<? extends DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		// TODO
	}

	@Override
	public NeuronsActivation getOutput() {
		return outputActivation;
	}

}
