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
import java.util.Optional;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adapter to wrap a DL4J IActivation instance so that it conforms to a ML4J DifferentiableActivationFunctionComponent interface.
 * 
 * Allows DL4J-specific activation functions to be used with a ML4J component graph or network.
 * 
 * @author Michael Lavelle
 *
 */
public class DL4JDifferentiableActivationFunctionComponentImpl extends DifferentiableActivationFunctionComponentBase implements DifferentiableActivationFunctionComponent {
	
	private static final Logger LOGGER = 
		      LoggerFactory.getLogger(DL4JDifferentiableActivationFunctionComponentImpl.class);
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private IActivation dl4jActivationFunction;
	private ActivationFunctionType activationFunctionType;
	private NeuronsActivationFeatureOrientation dl4jRequiredActivationOrientation;
	
	public DL4JDifferentiableActivationFunctionComponentImpl(Neurons neurons, IActivation dl4jActivationFunction, 
			ActivationFunctionType activationFunctionType, NeuronsActivationFeatureOrientation dl4jRequiredActivationOrientation) {
		super(neurons, activationFunctionType);
		this.dl4jActivationFunction = dl4jActivationFunction;
		this.activationFunctionType = activationFunctionType;
		this.dl4jRequiredActivationOrientation = dl4jRequiredActivationOrientation;
	}
	
	@Override
	public DifferentiableActivationFunctionComponentActivation forwardPropagate(NeuronsActivation neuronsActivation,
			NeuronsActivationContext context) {
		
		if (!supports().contains(neuronsActivation.getFeatureOrientation())) {
			throw new IllegalArgumentException("Input neurons activation format of:" + neuronsActivation.getFeatureOrientation() + " not supported");
		}
		if (optimisedFor().isPresent() && optimisedFor().get() != neuronsActivation.getFeatureOrientation()) {
			LOGGER.warn("Not using optimised input format");
		}
		
		NeuronsActivationFeatureOrientation dl4jActivationOrientation = dl4jRequiredActivationOrientation == null ? neuronsActivation.getFeatureOrientation() : dl4jRequiredActivationOrientation;
		
		INDArray inputNDArray = DL4JUtil.asNDArray(context.getMatrixFactory(), neuronsActivation, dl4jActivationOrientation);
		INDArray outputNDArray =  dl4jActivationFunction.getActivation(inputNDArray, context.isTrainingContext());
		NeuronsActivation outputActivation = DL4JUtil.fromNDArray(context.getMatrixFactory(), outputNDArray, dl4jActivationOrientation, neuronsActivation.getFeatureOrientation(), neurons);
		
		return new DL4JDifferentiableActivationFunctionComponentActivationImpl(context.getMatrixFactory(), 
				dl4jActivationFunction, activationFunctionType, neuronsActivation, inputNDArray, outputActivation, dl4jActivationOrientation);
	}
	
	@Override
	public NeuralComponentType<DifferentiableActivationFunctionComponent> getComponentType() {
			return NeuralComponentType.createSubType(NeuralComponentBaseType.ACTIVATION_FUNCTION, 
					activationFunctionType.getQualifiedId());
	}

	@Override
	public DifferentiableActivationFunctionComponent dup() {
		return new DL4JDifferentiableActivationFunctionComponentImpl(this.getInputNeurons(), dl4jActivationFunction, activationFunctionType, dl4jRequiredActivationOrientation);
	}

	@Override
	public List<NeuronsActivationFeatureOrientation> supports() {
		return Arrays.asList(NeuronsActivationFeatureOrientation.values());
	}

	@Override
	public Optional<NeuronsActivationFeatureOrientation> optimisedFor() {
		return dl4jRequiredActivationOrientation != null ? Optional.of(dl4jRequiredActivationOrientation) : Optional.empty();
	}
}
