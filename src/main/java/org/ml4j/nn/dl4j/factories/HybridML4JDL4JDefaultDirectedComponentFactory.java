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
package org.ml4j.nn.dl4j.factories;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.activationfunctions.DifferentiableActivationFunctionComponent;
import org.ml4j.nn.dl4j.activationfunctions.DL4JDifferentiableActivationFunctionComponentImpl;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.provider.Provider;
import org.ml4j.provider.enums.activationfunctions.ActivationFunctionTypeEnum;
import org.nd4j.linalg.activations.Activation;

/**
 * Extension of the default DefaultDirectedComponentFactoryImpl from ML4J which
 * uses DL4J components equivalents for some functionality.
 * 
 * Currently implemented so that activation functions from DL4J are used, while
 * other components are loaded from ML4J
 * 
 * @author Michael Lavelle
 */
public class HybridML4JDL4JDefaultDirectedComponentFactory extends DefaultDirectedComponentFactoryImpl {

	public HybridML4JDL4JDefaultDirectedComponentFactory(MatrixFactory matrixFactory, AxonsFactory axonsFactory) {
		super(matrixFactory, axonsFactory, null);
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons,
			DifferentiableActivationFunction differentiableActivationFunction) {
		return createDifferentiableActivationFunctionComponent(neurons,
				differentiableActivationFunction.getActivationFunctionType());
	}

	@Override
	public DifferentiableActivationFunctionComponent createDifferentiableActivationFunctionComponent(Neurons neurons,
			ActivationFunctionType activationFunctionType) {

		// Find the provider-agnostic ActivationFunctionTypeEnum from this ml4j-specific
		// type
		ActivationFunctionTypeEnum activationFunctionTypeEnum = ActivationFunctionTypeEnum
				.findByEnumValue(activationFunctionType.getBaseType()).orElseThrow(() -> new IllegalArgumentException(
						"Cannot find provider-agnostic activation function type for:" + activationFunctionType));

		// Get the dl4j equivalent enum
		Activation dl4jActivationFunctionType = activationFunctionTypeEnum.providedBy(Provider.DL4J)
				.getEnumAsType(Activation.class);

		NeuronsActivationFeatureOrientation requiredOrientation = activationFunctionType
				.getBaseType() == ActivationFunctionBaseType.SOFTMAX
						? NeuronsActivationFeatureOrientation.COLUMNS_SPAN_FEATURE_SET
						: null;
		return new DL4JDifferentiableActivationFunctionComponentImpl(neurons,
				dl4jActivationFunctionType.getActivationFunction(), activationFunctionType, requiredOrientation);

	}

}
