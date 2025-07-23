from typing import Dict, List, Any, Optional, Iterator
import logging
import time
from datetime import datetime
from bedrock_client import BedrockClient
from s3_client import S3Client
from prompt_manager import PromptManager
from config import Config

logger = logging.getLogger(__name__)

class LLMGrader:
    """
    Main LLM Grader class that orchestrates the entire grading process
    
    This class brings together S3 file management, prompt generation, and
    Bedrock LLM invocation to provide automated assignment grading.
    """
    
    def __init__(self, 
                 bedrock_client: Optional[BedrockClient] = None, 
                 s3_client: Optional[S3Client] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """
        Initialize the LLM Grader with client dependencies
        
        Args:
            bedrock_client: Bedrock client for LLM operations
            s3_client: S3 client for file operations  
            prompt_manager: Prompt manager for template operations
        """
        
        # Initialize clients (create new ones if not provided)
        self.bedrock_client = bedrock_client or BedrockClient()
        self.s3_client = s3_client or S3Client()
        self.prompt_manager = prompt_manager or PromptManager()
        
        # Test connections on initialization
        self.connection_status = self._test_all_connections()
        
        logger.info("LLMGrader initialized successfully")
        
        if self.connection_status['overall_healthy']:
            logger.info("✅ All connections healthy - ready for grading")
        else:
            logger.warning("⚠️ Some connections unhealthy - check system status")
    
    def _test_all_connections(self) -> Dict[str, Any]:
        """Test all client connections and return status"""
        try:
            logger.info("Testing all system connections...")
            
            # Test Bedrock connection
            bedrock_status = self.bedrock_client.test_connection()
            
            # Test S3 connection
            s3_status = self.s3_client.test_bucket_access()
            
            # Test prompt manager (check if templates exist)
            try:
                templates = self.prompt_manager.list_templates()
                prompt_status = {
                    'success': True,
                    'templates_available': len(templates),
                    'templates': templates
                }
            except Exception as e:
                prompt_status = {
                    'success': False,
                    'error': str(e),
                    'templates_available': 0
                }
            
            # Overall health check
            overall_healthy = (bedrock_status['success'] and 
                             s3_status['success'] and 
                             prompt_status['success'])
            
            status = {
                'bedrock': bedrock_status,
                's3': s3_status,
                'prompt_manager': prompt_status,
                'overall_healthy': overall_healthy,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Connection test completed - Overall healthy: {overall_healthy}")
            return status
            
        except Exception as e:
            logger.error(f"Error testing connections: {e}")
            return {
                'bedrock': {'success': False, 'error': 'Test failed'},
                's3': {'success': False, 'error': 'Test failed'},
                'prompt_manager': {'success': False, 'error': 'Test failed'},
                'overall_healthy': False,
                'error': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for debugging and monitoring
        
        Returns:
            Dictionary with system status and configuration
        """
        try:
            # Get fresh connection status
            current_connections = self._test_all_connections()
            
            # Get component information
            bedrock_info = self.bedrock_client.get_model_info()
            
            status = {
                'grader_status': 'operational' if current_connections['overall_healthy'] else 'degraded',
                'connections': current_connections,
                'bedrock_model': bedrock_info,
                's3_bucket': self.s3_client.bucket_name,
                'available_templates': self.prompt_manager.list_templates(),
                'config': {
                    'region': Config.AWS_REGION,
                    'max_tokens': Config.MAX_TOKENS,
                    'bucket_name': Config.S3_BUCKET_NAME
                },
                'last_updated': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'grader_status': 'error',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    def check_assignment_readiness(self, assignment_name: str) -> Dict[str, Any]:
        """
        Check if an assignment is ready for grading
        
        Args:
            assignment_name: Name of the assignment to check
            
        Returns:
            Dictionary with readiness status and details
        """
        try:
            logger.info(f"Checking assignment readiness: {assignment_name}")
            
            # Use S3 client's simple check
            check_result = self.s3_client.simple_assignment_check(assignment_name)
            
            # Add grader-specific analysis
            readiness = {
                'assignment_name': assignment_name,
                'ready_for_grading': check_result['ready_for_grading'],
                'check_details': check_result,
                'recommendations': [],
                'estimated_cost': None
            }
            
            # Add recommendations based on what's available
            if check_result['ready_for_grading']:
                readiness['recommendations'].append("✅ Assignment is ready for grading")
                
                if check_result['response_count'] == 0:
                    readiness['recommendations'].append("💡 Consider adding example student responses for better few-shot learning")
                elif check_result['response_count'] < 3:
                    readiness['recommendations'].append("💡 More example responses (3+) would improve grading consistency")
                
                # Estimate cost (very rough)
                try:
                    # Rough token estimation for cost prediction
                    estimated_prompt_tokens = 2000  # Base prompt
                    estimated_prompt_tokens += check_result['solution_count'] * 1000  # Solutions
                    estimated_output_tokens = 500  # Typical grading response
                    
                    # Rough cost calculation (Claude 3 Sonnet pricing)
                    input_cost = (estimated_prompt_tokens / 1000) * 0.003
                    output_cost = (estimated_output_tokens / 1000) * 0.015
                    total_cost = input_cost + output_cost
                    
                    readiness['estimated_cost'] = {
                        'estimated_input_tokens': estimated_prompt_tokens,
                        'estimated_output_tokens': estimated_output_tokens,
                        'estimated_cost_usd': round(total_cost, 4)
                    }
                except Exception as e:
                    logger.warning(f"Could not estimate cost: {e}")
            
            else:
                readiness['recommendations'].append(f"❌ Missing: {', '.join(check_result['missing'])}")
                readiness['recommendations'].append("📁 Upload missing files to S3 before grading")
            
            return readiness
            
        except Exception as e:
            logger.error(f"Error checking assignment readiness: {e}")
            return {
                'assignment_name': assignment_name,
                'ready_for_grading': False,
                'error': str(e)
            }
    
    def grade_assignment(self, 
                        assignment_name: str, 
                        target_response: str,
                        few_shot_examples: List[Dict[str, Any]],
                        template_name: str = "grading_prompt.j2",
                        additional_context: Optional[str] = None,
                        grading_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Grade a single assignment using the complete grading pipeline
        
        Args:
            assignment_name: Name of the assignment folder in S3
            target_response: Student response text to grade
            few_shot_examples: List of example responses with grades and feedback
            template_name: Template to use for prompt generation
            additional_context: Optional additional instructions or context
            grading_options: Optional grading parameters (temperature, max_tokens, etc.)
            
        Returns:
            Complete grading result dictionary
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Starting grading process for assignment: {assignment_name}")
            
            # Set default grading options
            options = grading_options or {}
            temperature = options.get('temperature', 0.1)  # Low for consistent grading
            max_tokens = options.get('max_tokens', Config.MAX_TOKENS)
            
            # Step 1: Check if assignment is ready
            logger.info("📋 Step 1: Checking assignment readiness...")
            readiness = self.check_assignment_readiness(assignment_name)
            
            if not readiness['ready_for_grading']:
                return {
                    'success': False,
                    'assignment_name': assignment_name,
                    'error': f"Assignment not ready for grading",
                    'readiness_check': readiness,
                    'stage_failed': 'readiness_check'
                }
            
            logger.info(f"✅ Assignment ready: {readiness['check_details']['solution_count']} solutions, "
                       f"{readiness['check_details']['response_count']} examples")
            
            # Step 2: Load assignment materials
            logger.info("📁 Step 2: Loading assignment materials from S3...")
            materials = self.s3_client.load_assignment_materials(assignment_name)
            
            if 'error' in materials:
                return {
                    'success': False,
                    'assignment_name': assignment_name,
                    'error': f"Failed to load assignment materials: {materials['error']}",
                    'readiness_check': readiness,
                    'stage_failed': 'material_loading'
                }
            
            logger.info(f"✅ Materials loaded: question ({len(materials['question'])} chars), "
                       f"{len(materials['solution_keys'])} solution keys")
            
            # Step 3: Prepare template variables
            logger.info("📝 Step 3: Preparing prompt template...")
            template_vars = {
                'case_study_question': materials['question'],
                'solution_keys': materials['solution_keys'],
                'few_shot_examples': few_shot_examples,
                'target_response': target_response
            }
            
            # Add additional context if provided
            if additional_context:
                template_vars['additional_context'] = additional_context
            
            # Step 4: Generate prompt using template
            logger.info(f"🔧 Step 4: Generating prompt using template: {template_name}")
            try:
                prompt = self.prompt_manager.generate_prompt(template_name, **template_vars)
            except Exception as e:
                return {
                    'success': False,
                    'assignment_name': assignment_name,
                    'error': f"Failed to generate prompt: {str(e)}",
                    'readiness_check': readiness,
                    'stage_failed': 'prompt_generation'
                }
            
            # Step 5: Validate prompt length
            logger.info("🔍 Step 5: Validating prompt...")
            validation = self.bedrock_client.validate_inputs(prompt, max_tokens=max_tokens)
            
            if not validation['valid']:
                return {
                    'success': False,
                    'assignment_name': assignment_name,
                    'error': f"Prompt validation failed: {'; '.join(validation['errors'])}",
                    'validation_details': validation,
                    'readiness_check': readiness,
                    'stage_failed': 'prompt_validation'
                }
            
            if validation['warnings']:
                logger.warning(f"Prompt warnings: {'; '.join(validation['warnings'])}")
            
            logger.info(f"✅ Prompt valid: ~{validation['estimated_input_tokens']} tokens")
            
            # Step 6: Invoke Bedrock model for grading
            logger.info("🤖 Step 6: Invoking Claude for grading...")
            bedrock_response = self.bedrock_client.invoke_model(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not bedrock_response['success']:
                return {
                    'success': False,
                    'assignment_name': assignment_name,
                    'error': f"Bedrock invocation failed: {bedrock_response['error']}",
                    'bedrock_details': bedrock_response,
                    'readiness_check': readiness,
                    'stage_failed': 'llm_invocation'
                }
            
            # Step 7: Compile comprehensive results
            total_time = time.time() - start_time
            
            logger.info(f"✅ Grading completed successfully in {total_time:.2f}s")
            
            result = {
                'success': True,
                'assignment_name': assignment_name,
                'template_used': template_name,
                'evaluation': bedrock_response['generated_text'],
                'grading_metadata': {
                    'total_processing_time': round(total_time, 2),
                    'materials_loaded': materials['metadata'],
                    'prompt_validation': {
                        'estimated_tokens': validation['estimated_input_tokens'],
                        'warnings': validation.get('warnings', [])
                    },
                    'model_usage': {
                        'model_id': bedrock_response['model_id'],
                        'input_tokens': bedrock_response['input_tokens'],
                        'output_tokens': bedrock_response['output_tokens'],
                        'total_tokens': bedrock_response['total_tokens'],
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'cost_estimate': bedrock_response.get('cost_estimate', {})
                    },
                    'grading_context': {
                        'few_shot_examples_count': len(few_shot_examples),
                        'additional_context_provided': additional_context is not None,
                        'solution_keys_used': len(materials['solution_keys']),
                        'target_response_length': len(target_response)
                    }
                },
                'readiness_check': readiness,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add the raw prompt for debugging if needed
            if options.get('include_prompt', False):
                result['debug_info'] = {
                    'generated_prompt': prompt,
                    'template_variables': list(template_vars.keys())
                }
            
            logger.info(f"📊 Usage: {bedrock_response['input_tokens']} + {bedrock_response['output_tokens']} tokens, "
                       f"Cost: ~${bedrock_response.get('cost_estimate', {}).get('total_cost', 0):.4f}")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Unexpected error in grading process: {e}")
            
            return {
                'success': False,
                'assignment_name': assignment_name,
                'error': f"Unexpected error: {str(e)}",
                'processing_time': round(total_time, 2),
                'stage_failed': 'unexpected_error',
                'timestamp': datetime.now().isoformat()
            }
    
    def grade_assignment_streaming(self, 
                                  assignment_name: str, 
                                  target_response: str,
                                  few_shot_examples: List[Dict[str, Any]],
                                  template_name: str = "grading_prompt.j2",
                                  additional_context: Optional[str] = None) -> Iterator[str]:
        """
        Grade an assignment with real-time streaming response
        
        Args:
            assignment_name: Name of the assignment folder in S3
            target_response: Student response text to grade
            few_shot_examples: List of example responses with grades and feedback
            template_name: Template to use for prompt generation
            additional_context: Optional additional instructions
            
        Yields:
            Streaming response chunks from the LLM
        """
        try:
            logger.info(f"🎯 Starting streaming grading for assignment: {assignment_name}")
            
            # Same setup as regular grading, but don't wait for full response
            yield "📋 Checking assignment readiness...\n"
            
            readiness = self.check_assignment_readiness(assignment_name)
            if not readiness['ready_for_grading']:
                yield f"❌ Assignment not ready: {readiness.get('error', 'Unknown error')}\n"
                return
            
            yield "✅ Assignment ready!\n\n"
            yield "📁 Loading materials from S3...\n"
            
            materials = self.s3_client.load_assignment_materials(assignment_name)
            if 'error' in materials:
                yield f"❌ Failed to load materials: {materials['error']}\n"
                return
            
            yield f"✅ Loaded: question + {len(materials['solution_keys'])} solution keys\n\n"
            yield "📝 Generating grading prompt...\n"
            
            # Prepare template and generate prompt
            template_vars = {
                'case_study_question': materials['question'],
                'solution_keys': materials['solution_keys'],
                'few_shot_examples': few_shot_examples,
                'target_response': target_response
            }
            
            if additional_context:
                template_vars['additional_context'] = additional_context
            
            prompt = self.prompt_manager.generate_prompt(template_name, **template_vars)
            
            yield "✅ Prompt generated!\n\n"
            yield "🤖 Claude is grading your assignment...\n"
            yield "=" * 50 + "\n\n"
            
            # Stream the actual grading response
            for chunk in self.bedrock_client.invoke_with_streaming(prompt, temperature=0.1):
                yield chunk
            
            yield "\n\n" + "=" * 50 + "\n"
            yield "✅ Grading completed!\n"
            
        except Exception as e:
            logger.error(f"Error in streaming grading: {e}")
            yield f"\n❌ Error: {str(e)}\n"
    
    def batch_grade_assignments(self, 
                               grading_requests: List[Dict[str, Any]],
                               template_name: str = "grading_prompt.j2") -> Dict[str, Any]:
        """
        Grade multiple assignments in a batch operation
        
        Args:
            grading_requests: List of grading request dictionaries, each containing:
                - assignment_name: str
                - target_response: str  
                - few_shot_examples: List[Dict]
                - additional_context: Optional[str]
            template_name: Template to use for all gradings
            
        Returns:
            Batch grading results with summary statistics
        """
        batch_start_time = time.time()
        
        logger.info(f"🔄 Starting batch grading of {len(grading_requests)} assignments")
        
        results = []
        successful_gradings = 0
        failed_gradings = 0
        total_tokens_used = 0
        total_cost = 0.0
        
        for i, request in enumerate(grading_requests, 1):
            assignment_name = request['assignment_name']
            
            logger.info(f"📝 Processing assignment {i}/{len(grading_requests)}: {assignment_name}")
            
            try:
                # Grade individual assignment
                result = self.grade_assignment(
                    assignment_name=assignment_name,
                    target_response=request['target_response'],
                    few_shot_examples=request['few_shot_examples'],
                    template_name=template_name,
                    additional_context=request.get('additional_context')
                )
                
                # Add batch information
                result['batch_info'] = {
                    'batch_index': i - 1,
                    'batch_total': len(grading_requests),
                    'processing_order': i
                }
                
                results.append(result)
                
                if result['success']:
                    successful_gradings += 1
                    
                    # Accumulate usage statistics
                    if 'grading_metadata' in result:
                        model_usage = result['grading_metadata'].get('model_usage', {})
                        total_tokens_used += model_usage.get('total_tokens', 0)
                        cost_estimate = model_usage.get('cost_estimate', {})
                        total_cost += cost_estimate.get('total_cost', 0)
                    
                    logger.info(f"✅ {assignment_name} graded successfully")
                else:
                    failed_gradings += 1
                    logger.error(f"❌ {assignment_name} failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                failed_gradings += 1
                logger.error(f"❌ Exception processing {assignment_name}: {e}")
                
                results.append({
                    'success': False,
                    'assignment_name': assignment_name,
                    'error': str(e),
                    'batch_info': {
                        'batch_index': i - 1,
                        'batch_total': len(grading_requests),
                        'processing_order': i
                    }
                })
            
            # Small delay between requests to be respectful to the API
            if i < len(grading_requests):
                time.sleep(0.5)
        
        # Calculate batch summary
        total_time = time.time() - batch_start_time
        
        batch_summary = {
            'total_requests': len(grading_requests),
            'successful_gradings': successful_gradings,
            'failed_gradings': failed_gradings,
            'success_rate': successful_gradings / len(grading_requests) if grading_requests else 0,
            'total_processing_time': round(total_time, 2),
            'average_time_per_assignment': round(total_time / len(grading_requests), 2) if grading_requests else 0,
            'total_tokens_used': total_tokens_used,
            'estimated_total_cost': round(total_cost, 4),
            'average_cost_per_assignment': round(total_cost / successful_gradings, 4) if successful_gradings > 0 else 0
        }
        
        logger.info(f"🎯 Batch grading completed: {successful_gradings}/{len(grading_requests)} successful "
                   f"({batch_summary['success_rate']:.1%}) in {total_time:.2f}s")
        
        return {
            'success': True,
            'batch_summary': batch_summary,
            'results': results,
            'template_used': template_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def preview_grading_prompt(self, 
                              assignment_name: str,
                              few_shot_examples: List[Dict[str, Any]],
                              template_name: str = "grading_prompt.j2",
                              sample_target_response: str = "[Sample student response for preview]",
                              additional_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Preview the grading prompt without actually invoking the LLM
        
        Args:
            assignment_name: Name of the assignment
            few_shot_examples: List of example responses
            template_name: Template to use
            sample_target_response: Sample response text for preview
            additional_context: Optional additional context
            
        Returns:
            Dictionary with prompt preview and metadata
        """
        try:
            logger.info(f"👀 Generating prompt preview for: {assignment_name}")
            
            # Check if assignment exists
            readiness = self.check_assignment_readiness(assignment_name)
            if not readiness['ready_for_grading']:
                return {
                    'success': False,
                    'error': f"Assignment not ready for preview: {readiness.get('error', 'Unknown error')}",
                    'readiness_check': readiness
                }
            
            # Load materials
            materials = self.s3_client.load_assignment_materials(assignment_name)
            if 'error' in materials:
                return {
                    'success': False,
                    'error': f"Could not load materials: {materials['error']}"
                }
            
            # Prepare template variables
            template_vars = {
                'case_study_question': materials['question'],
                'solution_keys': materials['solution_keys'],
                'few_shot_examples': few_shot_examples,
                'target_response': sample_target_response
            }
            
            if additional_context:
                template_vars['additional_context'] = additional_context
            
            # Generate prompt
            prompt = self.prompt_manager.generate_prompt(template_name, **template_vars)
            
            # Get prompt statistics
            token_estimate = self.bedrock_client.estimate_tokens(prompt)
            validation = self.bedrock_client.validate_inputs(prompt)
            
            logger.info(f"✅ Prompt preview generated: ~{token_estimate} tokens")
            
            return {
                'success': True,
                'assignment_name': assignment_name,
                'template_used': template_name,
                'prompt_preview': prompt,
                'prompt_statistics': {
                    'estimated_tokens': token_estimate,
                    'character_count': len(prompt),
                    'line_count': prompt.count('\n') + 1,
                    'validation': validation
                },
                'materials_info': {
                    'question_length': len(materials['question']),
                    'solution_keys_count': len(materials['solution_keys']),
                    'few_shot_examples_count': len(few_shot_examples)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating prompt preview: {e}")
            return {
                'success': False,
                'error': str(e),
                'assignment_name': assignment_name
            }
    
    def get_assignment_info(self, assignment_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about an assignment
        
        Args:
            assignment_name: Name of the assignment
            
        Returns:
            Dictionary with assignment information and analysis
        """
        try:
            logger.info(f"📊 Getting assignment info: {assignment_name}")
            
            # Get basic structure
            structure = self.s3_client.get_assignment_structure(assignment_name)
            
            # Get readiness status
            readiness = self.check_assignment_readiness(assignment_name)
            
            # Compile comprehensive info
            info = {
                'assignment_name': assignment_name,
                'structure': structure,
                'readiness': readiness,
                's3_info': {
                    'bucket_name': self.s3_client.bucket_name,
                    'region': self.s3_client.region_name
                },
                'grader_info': {
                    'templates_available': self.prompt_manager.list_templates(),
                    'model_info': self.bedrock_client.get_model_info()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'success': True,
                'assignment_info': info
            }
            
        except Exception as e:
            logger.error(f"Error getting assignment info: {e}")
            return {
                'success': False,
                'error': str(e),
                'assignment_name': assignment_name
            }
    
    def __str__(self) -> str:
        """String representation of the grader"""
        return f"LLMGrader(bucket={self.s3_client.bucket_name}, model={self.bedrock_client.model_id})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"LLMGrader("
                f"s3_bucket='{self.s3_client.bucket_name}', "
                f"bedrock_model='{self.bedrock_client.model_id}', "
                f"region='{self.bedrock_client.region_name}')")