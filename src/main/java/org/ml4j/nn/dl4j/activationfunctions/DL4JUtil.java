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
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Utilities for mapping between ML4J NeuronsActivation and Matrix instances and 
 * ND4J's INDArrays.
 * 
 * @author Michael Lavelle
 */
public class DL4JUtil {
	
	private static float[] getData(INDArray ndArray) {
		return ndArray.data().asFloat();
	}

	public static NeuronsActivation fromNDArray(MatrixFactory matrixFactory, INDArray ndArray, NeuronsActivationFeatureOrientation sourceOrientation, NeuronsActivationFeatureOrientation targetOrientation, Neurons neurons, boolean imageActivation) {
		
		float[] rowsByRowsArray = getData(ndArray);
		
		Matrix matrix = sourceOrientation.equals(targetOrientation) ? 
				matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray) 
				: matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray).transpose();
	
		
		NeuronsActivation neuronsActivation = new NeuronsActivationImpl(matrix,
				targetOrientation);
		if (imageActivation) {
			return neuronsActivation.asImageNeuronsActivation((Neurons3D)neurons);
		} else {
			return neuronsActivation;
		}
	}
	
	public static Matrix fromNDArrayToActivationMatrix(MatrixFactory matrixFactory, INDArray ndArray, NeuronsActivationFeatureOrientation sourceOrientation, NeuronsActivationFeatureOrientation targetOrientation) {
		float[] rowsByRowsArray = getData(ndArray);
		
		Matrix matrix = sourceOrientation.equals(targetOrientation) ? 
				matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray) 
				: matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray).transpose();
	
		
		return matrix;
	}
	
	public static INDArray asNDArray(MatrixFactory matrixFactory, NeuronsActivation neuronsActivation, NeuronsActivationFeatureOrientation targetOrientation) {
		Matrix matrix = targetOrientation == neuronsActivation.getFeatureOrientation() ? neuronsActivation.getActivations(matrixFactory) : neuronsActivation.getActivations(matrixFactory).transpose();
		return Nd4j.create(matrix.getRowByRowArray(), new int[] { matrix.getRows(), matrix.getColumns() });
	}
	
	
	
	public static INDArray asNDArrayForWeights(MatrixFactory matrixFactory, Matrix matrix, boolean transpose) {
		matrix = transpose ? matrix.transpose() : matrix;
		return Nd4j.create(matrix.getRowByRowArray(), new int[] { matrix.getRows(), matrix.getColumns() });
	}
	
	public static INDArray asNDArrayForBias(MatrixFactory matrixFactory, Matrix matrix, boolean transpose) {
		matrix = transpose ? matrix.transpose() : matrix;
		return Nd4j.create(matrix.getRowByRowArray(), new int[] { matrix.getRows(), matrix.getColumns() });
	}
	
	public static Matrix fromNDArrayToBiasMatrix(MatrixFactory matrixFactory, INDArray ndArray, boolean transpose) {
		float[] rowsByRowsArray = getData(ndArray);
		if (transpose) {
		return matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray).transpose();
		} else {
			return matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray);

		}
	}
	
	public static Matrix fromNDArrayToWeightsMatrix(MatrixFactory matrixFactory, INDArray ndArray, boolean transpose) {
		float[] rowsByRowsArray = getData(ndArray);
		if (transpose) {
		return matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray).transpose();
		} else {
			return matrixFactory.createMatrixFromRowsByRowsArray(ndArray.rows(), ndArray.columns(), rowsByRowsArray);

		}
	}
	
}
