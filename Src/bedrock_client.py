import boto3
import json
import logging
import time
from typing import Dict, Any, Optional, Iterator, Union
from botocore.exceptions import ClientError, BotoCoreError
from config import Config

logger = logging.getLogger(__name__)

class BedrockClient:
    """
    Enhanced Bedrock client for LLM grading operations
    
    This client handles all interactions with AWS Bedrock, specifically optimized
    for Claude models used in academic grading scenarios.
    """
    
    def __init__(self, region_name: Optional[str] = None, model_id: Optional[str] = None):
        """
        Initialize Bedrock client
        
        Args:
            region_name: AWS region (defaults to Config.AWS_REGION)
            model_id: Bedrock model ID (defaults to Config.BEDROCK_MODEL_ID)
        """
        self.region_name = region_name or Config.AWS_REGION
        self.model_id = model_id or Config.BEDROCK_MODEL_ID
        self.max_tokens = Config.MAX_TOKENS
        
        # Model-specific configurations
        self.model_config = self._get_model_config()
        
        # Initialize boto3 client
        try:
            self.client = boto3.client('bedrock-runtime', region_name=self.region_name)
            logger.info(f"Bedrock client initialized successfully")
            logger.info(f"Region: {self.region_name}, Model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise ConnectionError(f"Cannot connect to Bedrock: {e}")
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        configs = {
            # Claude 3 Sonnet
            'anthropic.claude-3-sonnet-20240229-v1:0': {
                'max_input_tokens': 200000,
                'max_output_tokens': 4096,
                'supports_system_prompt': True,
                'supports_streaming': True,
                'default_temperature': 0.1,  # Low for consistent grading
                'anthropic_version': 'bedrock-2023-05-31'
            },
            # Claude 3 Haiku
            'anthropic.claude-3-haiku-20240307-v1:0': {
                'max_input_tokens': 200000,
                'max_output_tokens': 4096,
                'supports_system_prompt': True,
                'supports_streaming': True,
                'default_temperature': 0.1,
                'anthropic_version': 'bedrock-2023-05-31'
            },
            # Claude 3 Opus
            'anthropic.claude-3-opus-20240229-v1:0': {
                'max_input_tokens': 200000,
                'max_output_tokens': 4096,
                'supports_system_prompt': True,
                'supports_streaming': True,
                'default_temperature': 0.1,
                'anthropic_version': 'bedrock-2023-05-31'
            }
        }
        
        return configs.get(self.model_id, {
            'max_input_tokens': 100000,
            'max_output_tokens': 4096,
            'supports_system_prompt': False,
            'supports_streaming': False,
            'default_temperature': 0.3,
            'anthropic_version': 'bedrock-2023-05-31'
        })
    
    def invoke_model(self, 
                    prompt: str, 
                    system_prompt: Optional[str] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Invoke the Bedrock model with given prompt
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt (if model supports it)
            temperature: Model temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Dict containing the model response and metadata
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            validation_result = self.validate_inputs(prompt, system_prompt, max_tokens)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'validation': validation_result
                }
            
            # Prepare request parameters
            request_params = self._prepare_request_params(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
            
            logger.info(f"Invoking model: {self.model_id}")
            logger.debug(f"Request parameters: {json.dumps({k: v for k, v in request_params.items() if k != 'body'}, indent=2)}")
            
            # Make the API call
            response = self.client.invoke_model(**request_params)
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract response data
            result = self._parse_response(response_body, start_time)
            
            logger.info(f"Model invocation successful. "
                       f"Input tokens: {result['input_tokens']}, "
                       f"Output tokens: {result['output_tokens']}, "
                       f"Duration: {result['duration']:.2f}s")
            
            return result
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"AWS ClientError: {error_code} - {error_msg}")
            
            return {
                'success': False,
                'error': f"AWS Error ({error_code}): {error_msg}",
                'error_type': 'client_error',
                'error_code': error_code,
                'duration': time.time() - start_time
            }
            
        except BotoCoreError as e:
            logger.error(f"BotoCoreError: {e}")
            return {
                'success': False,
                'error': f"Connection error: {str(e)}",
                'error_type': 'connection_error',
                'duration': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Unexpected error invoking Bedrock model: {e}")
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'error_type': 'unexpected_error',
                'duration': time.time() - start_time
            }
    
    def _prepare_request_params(self, 
                               prompt: str,
                               system_prompt: Optional[str] = None,
                               temperature: Optional[float] = None,
                               max_tokens: Optional[int] = None,
                               top_p: Optional[float] = None,
                               top_k: Optional[int] = None) -> Dict[str, Any]:
        """Prepare request parameters for the model"""
        
        # Use defaults if not provided
        temperature = temperature if temperature is not None else self.model_config['default_temperature']
        max_tokens = max_tokens or self.max_tokens
        
        # Prepare messages
        messages = []
        
        # Add system prompt if supported and provided
        if system_prompt and self.model_config['supports_system_prompt']:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        elif system_prompt and not self.model_config['supports_system_prompt']:
            # Prepend system prompt to user message if system prompts not supported
            prompt = f"System: {system_prompt}\n\nHuman: {prompt}"
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare request body
        request_body = {
            "anthropic_version": self.model_config['anthropic_version'],
            "max_tokens": min(max_tokens, self.model_config['max_output_tokens']),
            "temperature": max(0.0, min(1.0, temperature)),  # Clamp between 0 and 1
            "messages": messages
        }
        
        # Add optional parameters
        if top_p is not None:
            request_body["top_p"] = max(0.0, min(1.0, top_p))
        
        if top_k is not None:
            request_body["top_k"] = max(1, int(top_k))
        
        return {
            "modelId": self.model_id,
            "body": json.dumps(request_body)
        }
    
    def _parse_response(self, response_body: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Parse the model response"""
        
        # Extract content
        content = response_body.get('content', [])
        if not content:
            raise ValueError("No content in response")
        
        generated_text = content[0].get('text', '')
        
        # Extract usage statistics
        usage = response_body.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Calculate costs (approximate)
        cost_info = self._calculate_costs(input_tokens, output_tokens)
        
        return {
            'success': True,
            'generated_text': generated_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'duration': duration,
            'model_id': self.model_id,
            'cost_estimate': cost_info,
            'raw_response': response_body
        }
    
    def _calculate_costs(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Calculate approximate costs based on current AWS pricing"""
        
        # Approximate pricing (as of 2024 - check AWS for current rates)
        pricing = {
            'anthropic.claude-3-sonnet-20240229-v1:0': {
                'input': 0.003 / 1000,   # $0.003 per 1K input tokens
                'output': 0.015 / 1000   # $0.015 per 1K output tokens
            },
            'anthropic.claude-3-haiku-20240307-v1:0': {
                'input': 0.00025 / 1000,  # $0.00025 per 1K input tokens
                'output': 0.00125 / 1000  # $0.00125 per 1K output tokens
            },
            'anthropic.claude-3-opus-20240229-v1:0': {
                'input': 0.015 / 1000,   # $0.015 per 1K input tokens
                'output': 0.075 / 1000   # $0.075 per 1K output tokens
            }
        }
        
        model_pricing = pricing.get(self.model_id, {'input': 0.001/1000, 'output': 0.002/1000})
        
        input_cost = input_tokens * model_pricing['input']
        output_cost = output_tokens * model_pricing['output']
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6),
            'currency': 'USD'
        }
    
    def invoke_with_streaming(self, 
                             prompt: str,
                             system_prompt: Optional[str] = None,
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None) -> Iterator[str]:
        """
        Invoke model with streaming response for real-time output
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Streaming response chunks as strings
        """
        try:
            if not self.model_config['supports_streaming']:
                logger.warning(f"Model {self.model_id} does not support streaming")
                # Fallback to regular invocation
                result = self.invoke_model(prompt, system_prompt, temperature, max_tokens)
                if result['success']:
                    yield result['generated_text']
                else:
                    yield f"Error: {result['error']}"
                return
            
            # Prepare request parameters
            request_params = self._prepare_request_params(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"Starting streaming invocation for model: {self.model_id}")
            
            # Make streaming request
            response = self.client.invoke_model_with_response_stream(**request_params)
            
            # Process streaming response
            for event in response['body']:
                chunk = json.loads(event['chunk']['bytes'])
                
                if chunk['type'] == 'content_block_delta':
                    delta_text = chunk['delta'].get('text', '')
                    if delta_text:
                        yield delta_text
                        
                elif chunk['type'] == 'message_stop':
                    logger.info("Streaming completed successfully")
                    break
                    
                elif chunk['type'] == 'error':
                    error_msg = chunk.get('error', {}).get('message', 'Unknown streaming error')
                    logger.error(f"Streaming error: {error_msg}")
                    yield f"\n[Error: {error_msg}]"
                    break
                    
        except Exception as e:
            logger.error(f"Error in streaming invocation: {e}")
            yield f"\n[Streaming Error: {str(e)}]"
    
    def validate_inputs(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Validate input parameters"""
        
        errors = []
        warnings = []
        
        # Check prompt
        if not prompt or not prompt.strip():
            errors.append("Prompt cannot be empty")
        
        # Estimate token count
        total_text = prompt
        if system_prompt:
            total_text += system_prompt
            
        estimated_tokens = self.estimate_tokens(total_text)
        max_input_tokens = self.model_config['max_input_tokens']
        
        if estimated_tokens > max_input_tokens:
            errors.append(f"Estimated input tokens ({estimated_tokens}) exceed model limit ({max_input_tokens})")
        elif estimated_tokens > max_input_tokens * 0.9:
            warnings.append(f"Input is very long ({estimated_tokens} tokens). Consider shortening.")
        
        # Check max_tokens parameter
        if max_tokens and max_tokens > self.model_config['max_output_tokens']:
            warnings.append(f"Requested output tokens ({max_tokens}) exceed model limit ({self.model_config['max_output_tokens']})")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'estimated_input_tokens': estimated_tokens,
            'max_input_tokens': max_input_tokens,
            'recommendation': 'Consider shortening the prompt' if estimated_tokens > max_input_tokens * 0.8 else 'Input length is acceptable'
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of tokens in text
        Claude typically uses ~3.5-4 characters per token for English text
        """
        if not text:
            return 0
        
        # More accurate estimation based on text characteristics
        char_count = len(text)
        
        # Adjust based on text type
        if text.count(' ') > char_count * 0.15:  # Lots of spaces (normal text)
            chars_per_token = 4.0
        elif text.count('\n') > char_count * 0.05:  # Lots of newlines (formatted text)
            chars_per_token = 3.5
        else:  # Dense text
            chars_per_token = 3.8
        
        return int(char_count / chars_per_token)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Bedrock connection with a simple prompt"""
        test_prompt = "Hello! Please respond with exactly: 'Connection test successful'"
        
        try:
            logger.info("Testing Bedrock connection...")
            result = self.invoke_model(
                prompt=test_prompt,
                max_tokens=50,
                temperature=0.0
            )
            
            if result['success']:
                response_text = result['generated_text'].strip()
                connection_works = 'connection test successful' in response_text.lower()
                
                logger.info("Bedrock connection test completed successfully")
                return {
                    'success': True,
                    'connection_works': connection_works,
                    'model_id': self.model_id,
                    'region': self.region_name,
                    'response': response_text,
                    'latency': result['duration'],
                    'test_tokens': result['total_tokens']
                }
            else:
                logger.error(f"Bedrock connection test failed: {result['error']}")
                return {
                    'success': False,
                    'connection_works': False,
                    'model_id': self.model_id,
                    'region': self.region_name,
                    'error': result['error'],
                    'error_type': result.get('error_type', 'unknown')
                }
                
        except Exception as e:
            logger.error(f"Bedrock connection test error: {e}")
            return {
                'success': False,
                'connection_works': False,
                'model_id': self.model_id,
                'region': self.region_name,
                'error': str(e),
                'error_type': 'connection_error'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current model"""
        return {
            'model_id': self.model_id,
            'region': self.region_name,
            'max_tokens': self.max_tokens,
            'model_config': self.model_config,
            'supported_features': [
                'text_generation',
                'streaming' if self.model_config['supports_streaming'] else None,
                'system_prompts' if self.model_config['supports_system_prompt'] else None,
                'temperature_control',
                'top_p_sampling',
                'top_k_sampling'
            ],
            'client_version': boto3.__version__
        }
    
    def batch_invoke(self, 
                    prompts: list,
                    system_prompt: Optional[str] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None) -> list:
        """
        Process multiple prompts in sequence
        
        Args:
            prompts: List of prompt strings
            system_prompt: Optional system prompt for all requests
            temperature: Model temperature
            max_tokens: Maximum tokens per response
            
        Returns:
            List of response dictionaries
        """
        results = []
        total_prompts = len(prompts)
        
        logger.info(f"Starting batch processing of {total_prompts} prompts")
        
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Processing prompt {i}/{total_prompts}")
            
            result = self.invoke_model(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result['batch_index'] = i - 1
            result['batch_total'] = total_prompts
            results.append(result)
            
            # Brief pause between requests to be respectful
            if i < total_prompts:
                time.sleep(0.1)
        
        logger.info(f"Batch processing completed. {len(results)} results generated")
        return results
    
    def get_usage_summary(self, results: list) -> Dict[str, Any]:
        """Generate usage summary from multiple results"""
        if not results:
            return {'error': 'No results provided'}
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful results to summarize'}
        
        total_input_tokens = sum(r.get('input_tokens', 0) for r in successful_results)
        total_output_tokens = sum(r.get('output_tokens', 0) for r in successful_results)
        total_duration = sum(r.get('duration', 0) for r in successful_results)
        total_cost = sum(r.get('cost_estimate', {}).get('total_cost', 0) for r in successful_results)
        
        return {
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'failed_requests': len(results) - len(successful_results),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'total_duration': round(total_duration, 2),
            'average_duration': round(total_duration / len(successful_results), 2),
            'estimated_total_cost': round(total_cost, 6),
            'average_cost_per_request': round(total_cost / len(successful_results), 6) if successful_results else 0
        }
    
    def __str__(self) -> str:
        """String representation of the client"""
        return f"BedrockClient(model={self.model_id}, region={self.region_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"BedrockClient(model_id='{self.model_id}', region='{self.region_name}', max_tokens={self.max_tokens})"