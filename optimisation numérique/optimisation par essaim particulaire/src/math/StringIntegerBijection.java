/**
 * Copyright (c) 2019 Hamza JAFFALI.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package math;

import java.math.BigInteger;
import java.util.Arrays;

public class StringIntegerBijection {

	//Defining all allowed characters for writing the message
	public static char[] tabCorrespondance = new char[]{'.','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
	public static int tabSize = tabCorrespondance.length;
	
	/** Return a unique integer encoding the String in parameter
	 * 
	 * @param message the String to be encoded
	 * @return the integer related to the String in parameter
	 */
	public static BigInteger StringToInteger (String message) {
		// Formatting the String in parameter to UpperCase
		String upperCaseMessage = message.toUpperCase();
		BigInteger number = new BigInteger("0");
		for (int i=0; i<message.length(); i++) {
			// Getting the i-th character of the message
			char actualChar = upperCaseMessage.charAt(i);
			// Searching for this character in the array of allowed characters
			int index = Arrays.binarySearch(tabCorrespondance, actualChar);
			number = number.add(BigInteger.valueOf(((long) Math.pow(tabSize, message.length()-1-i))*index));
		}
		
		return number;
	}
	
	/** Return the unique String associated to the integer in parameter
	 * 
	 * @param number the integer encoding a String
	 * @return the String encoded by the integer in parameter
	 */
	public static String IntegerToString (BigInteger number) {
		String message = "";
		BigInteger quotient = number.divide(BigInteger.valueOf(tabSize));
		BigInteger remainder = number.remainder(BigInteger.valueOf(tabSize));
		message = tabCorrespondance[remainder.intValue()] + message;
		while (quotient.compareTo(BigInteger.valueOf(tabSize))>0) {
			remainder = quotient.remainder(BigInteger.valueOf(tabSize));
			quotient = quotient.divide(BigInteger.valueOf(tabSize));
			message = tabCorrespondance[remainder.intValue()] + message;
		}
		message = tabCorrespondance[quotient.intValue()] + message;
		
		return message;
	}
	
	
	public static void main(String[] args) {
		
		// Some tests and demos of the functions		
		System.out.println(StringToInteger("RSA"));
		System.out.println(IntegerToString(new BigInteger("263500499")));
		System.out.println(IntegerToString(new BigInteger("276910182")));
		System.out.println(IntegerToString(new BigInteger("451301342")));
		System.out.println(IntegerToString(new BigInteger("358488")));
	}
}
