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
import java.util.HashSet;
import java.util.Optional;
import java.util.Set;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentBase;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.FlatFeaturesFormat;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adapter to wrap a DL4J IActivation instance so that it conforms to a ML4J
 * DifferentiableActivationFunctionComponent interface.
 * 
 * Allows DL4J-specific activation functions to be used with a ML4J component
 * graph or network.
 * 
 * @author Michael Lavelle
 *
 */
public class DL4JDifferentiableActivationFunctionComponentImpl extends DifferentiableActivationFunctionComponentBase
		implements DifferentiableActivationFunctionComponent {

	private static final Logger LOGGER = LoggerFactory
			.getLogger(DL4JDifferentiableActivationFunctionComponentImpl.class);

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private IActivation dl4jActivationFunction;
	private ActivationFunctionType activationFunctionType;
	private NeuronsActivationFeatureOrientation dl4jRequiredActivationOrientation;

	public DL4JDifferentiableActivationFunctionComponentImpl(String name,Neurons neurons, IActivation dl4jActivationFunction,
			ActivationFunctionType activationFunctionType,
			NeuronsActivationFeatureOrientation dl4jRequiredActivationOrientation) {
		super(name, neurons, activationFunctionType);
		this.dl4jActivationFunction = dl4jActivationFunction;
		this.activationFunctionType = activationFunctionType;
		this.dl4jRequiredActivationOrientation = dl4jRequiredActivationOrientation;
	}

	@Override
	public DifferentiableActivationFunctionComponentActivation forwardPropagate(NeuronsActivation neuronsActivation,
			NeuronsActivationContext context) {
		
		if (!isSupported(neuronsActivation.getFormat())) {
			throw new IllegalArgumentException("Input neurons activation format of:"
					+ neuronsActivation.getFormat() + " not supported");
		}
		if (optimisedFor().isPresent() && !optimisedFor().get().equals(neuronsActivation.getFormat())) {
			LOGGER.warn("Not using optimised input format");
		}

		NeuronsActivationFeatureOrientation dl4jActivationOrientation = dl4jRequiredActivationOrientation == null
				? neuronsActivation.getFeatureOrientation()
				: dl4jRequiredActivationOrientation;

		INDArray inputNDArray = DL4JUtil.asNDArray(context.getMatrixFactory(), neuronsActivation,
				dl4jActivationOrientation);
		INDArray outputNDArray = dl4jActivationFunction.getActivation(inputNDArray, context.isTrainingContext());
		NeuronsActivation outputActivation = DL4JUtil.fromNDArray(context.getMatrixFactory(), outputNDArray,
				dl4jActivationOrientation, neuronsActivation.getFormat(), neurons);

		return new DL4JDifferentiableActivationFunctionComponentActivationImpl(context.getMatrixFactory(),
				dl4jActivationFunction, activationFunctionType, neuronsActivation, inputNDArray, outputActivation,
				dl4jActivationOrientation);
	}

	@Override
	public NeuralComponentType getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentBaseType.ACTIVATION_FUNCTION,
				activationFunctionType.getQualifiedId());
	}

	@Override
	public DifferentiableActivationFunctionComponent dup(DirectedComponentFactory directedComponentFactory) {
		return new DL4JDifferentiableActivationFunctionComponentImpl(name, this.getInputNeurons(), dl4jActivationFunction,
				activationFunctionType, dl4jRequiredActivationOrientation);
	}
	
	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return dl4jRequiredActivationOrientation != null ? Optional.of(new NeuronsActivationFormat<>(dl4jRequiredActivationOrientation, 
				new FlatFeaturesFormat(), Arrays.asList(Dimension.EXAMPLE)))
				: Optional.empty();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return true;
	}
	
	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		return allComponentsIncludingThis;
	}

}
