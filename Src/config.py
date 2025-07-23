import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Configuration management for the LLM Grader system
    
    This class centralizes all configuration settings and provides
    validation, environment variable loading, and default values.
    """
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'your-assignments-bucket')
    
    # Bedrock Configuration  
    BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4000'))
    
    # Template Configuration
    TEMPLATES_DIR = os.getenv('TEMPLATES_DIR', 'templates')
    DEFAULT_TEMPLATE = os.getenv('DEFAULT_TEMPLATE', 'grading_prompt.j2')
    
    # Output Configuration
    RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    
    # Grading Configuration
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.1'))
    DEFAULT_TOP_P = float(os.getenv('DEFAULT_TOP_P', '0.9'))
    
    # File Processing Configuration
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    SUPPORTED_FILE_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc']
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Model-specific configurations
    MODEL_CONFIGS = {
        'anthropic.claude-3-sonnet-20240229-v1:0': {
            'max_input_tokens': 200000,
            'max_output_tokens': 4096,
            'cost_per_1k_input': 0.003,
            'cost_per_1k_output': 0.015,
            'description': 'Claude 3 Sonnet - Balanced performance and cost'
        },
        'anthropic.claude-3-haiku-20240307-v1:0': {
            'max_input_tokens': 200000,
            'max_output_tokens': 4096,
            'cost_per_1k_input': 0.00025,
            'cost_per_1k_output': 0.00125,
            'description': 'Claude 3 Haiku - Fast and economical'
        },
        'anthropic.claude-3-opus-20240229-v1:0': {
            'max_input_tokens': 200000,
            'max_output_tokens': 4096,
            'cost_per_1k_input': 0.015,
            'cost_per_1k_output': 0.075,
            'description': 'Claude 3 Opus - Highest capability'
        }
    }
    
    # Grading scale configuration
    GRADING_SCALE = {
        'A': {'min': 90, 'max': 100, 'description': 'Excellent'},
        'B': {'min': 80, 'max': 89, 'description': 'Good'}, 
        'C': {'min': 70, 'max': 79, 'description': 'Average'},
        'D': {'min': 60, 'max': 69, 'description': 'Below Average'},
        'F': {'min': 0, 'max': 59, 'description': 'Failing'}
    }
    
    # Assignment structure configuration
    REQUIRED_FOLDERS = ['question', 'solution_keys']
    OPTIONAL_FOLDERS = ['student_responses']
    
    @classmethod
    def get_model_config(cls, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        
        Args:
            model_id: Model identifier (uses default if None)
            
        Returns:
            Model configuration dictionary
        """
        model_id = model_id or cls.BEDROCK_MODEL_ID
        
        config = cls.MODEL_CONFIGS.get(model_id, {
            'max_input_tokens': 100000,
            'max_output_tokens': 4000,
            'cost_per_1k_input': 0.001,
            'cost_per_1k_output': 0.002,
            'description': 'Unknown model - using defaults'
        })
        
        return config
    
    @classmethod
    def get_few_shot_examples(cls) -> List[Dict[str, str]]:
        """
        Get default few-shot examples for grading
        
        Returns:
            List of example grading cases
        """
        return [
            {
                'grade': 'A (95)',
                'response': '''This response demonstrates exceptional analytical depth and strategic thinking. 
The student provides comprehensive market analysis, detailed financial projections, and well-reasoned 
recommendations. The analysis incorporates multiple business frameworks effectively and shows sophisticated 
understanding of competitive dynamics. The writing is clear, well-structured, and professionally presented.''',
                'feedback': '''Excellent work! This response shows mastery of strategic analysis concepts. 
Your market research is thorough, financial modeling is accurate, and recommendations are well-supported. 
The integration of course concepts with real-world application is particularly strong. Minor suggestion: 
consider adding more specific implementation timelines for your recommendations.'''
            },
            {
                'grade': 'B (85)',
                'response': '''This response shows good understanding of the core concepts and provides solid analysis. 
The student identifies key issues and offers reasonable solutions, though some areas lack depth. The financial 
analysis is generally accurate but could be more comprehensive. The reasoning is sound but would benefit from 
more detailed supporting evidence and consideration of alternative approaches.''',
                'feedback': '''Good work overall! You demonstrate solid grasp of the fundamental concepts and your 
analysis addresses the main issues effectively. To improve: develop your financial analysis further, consider 
more alternative scenarios, and strengthen your arguments with additional supporting evidence. Your conclusions 
are reasonable but could be more thoroughly justified.'''
            },
            {
                'grade': 'C (75)',
                'response': '''This response shows basic understanding of the topic but lacks analytical depth. 
The student identifies some relevant issues but fails to develop comprehensive solutions. The analysis is 
superficial and relies heavily on generalizations rather than specific evidence. The financial considerations 
are minimal and the recommendations lack clear justification or implementation details.''',
                'feedback': '''Your response demonstrates basic understanding but needs significant development. 
Focus on deeper analysis rather than surface-level observations. Include more specific examples, develop your 
financial analysis, and provide clearer justification for your recommendations. Consider multiple perspectives 
and potential risks in your analysis.'''
            }
        ]
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration settings and environment
        
        Returns:
            Validation results with errors and warnings
        """
        errors = []
        warnings = []
        
        # Required configuration checks
        if not cls.S3_BUCKET_NAME or cls.S3_BUCKET_NAME == 'your-assignments-bucket':
            errors.append("S3_BUCKET_NAME must be set to your actual bucket name")
        
        if not cls.AWS_REGION:
            errors.append("AWS_REGION is required")
        
        if not cls.BEDROCK_MODEL_ID:
            errors.append("BEDROCK_MODEL_ID is required")
        
        # Validate numeric values
        try:
            if cls.MAX_TOKENS <= 0 or cls.MAX_TOKENS > 8192:
                warnings.append(f"MAX_TOKENS ({cls.MAX_TOKENS}) should be between 1 and 8192")
        except (ValueError, TypeError):
            errors.append("MAX_TOKENS must be a valid integer")
        
        try:
            if cls.DEFAULT_TEMPERATURE < 0 or cls.DEFAULT_TEMPERATURE > 1:
                warnings.append(f"DEFAULT_TEMPERATURE ({cls.DEFAULT_TEMPERATURE}) should be between 0 and 1")
        except (ValueError, TypeError):
            errors.append("DEFAULT_TEMPERATURE must be a valid float between 0 and 1")
        
        try:
            if cls.MAX_FILE_SIZE_MB <= 0 or cls.MAX_FILE_SIZE_MB > 1000:
                warnings.append(f"MAX_FILE_SIZE_MB ({cls.MAX_FILE_SIZE_MB}) seems unusually large")
        except (ValueError, TypeError):
            errors.append("MAX_FILE_SIZE_MB must be a valid positive integer")
        
        # Check model configuration
        if cls.BEDROCK_MODEL_ID not in cls.MODEL_CONFIGS:
            warnings.append(f"Model '{cls.BEDROCK_MODEL_ID}' not in known configurations - using defaults")
        
        # Check directory paths
        for dir_config in [cls.RESULTS_DIR, cls.LOGS_DIR, cls.TEMPLATES_DIR]:
            if not dir_config:
                warnings.append(f"Directory path is empty: {dir_config}")
        
        # Log level validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            warnings.append(f"LOG_LEVEL '{cls.LOG_LEVEL}' not in {valid_log_levels}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config_summary': cls.get_config_summary()
        }
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of current configuration
        
        Returns:
            Configuration summary dictionary
        """
        model_config = cls.get_model_config()
        
        return {
            'aws': {
                'region': cls.AWS_REGION,
                's3_bucket': cls.S3_BUCKET_NAME,
                'bedrock_model': cls.BEDROCK_MODEL_ID
            },
            'grading': {
                'max_tokens': cls.MAX_TOKENS,
                'temperature': cls.DEFAULT_TEMPERATURE,
                'top_p': cls.DEFAULT_TOP_P
            },
            'files': {
                'max_size_mb': cls.MAX_FILE_SIZE_MB,
                'supported_extensions': cls.SUPPORTED_FILE_EXTENSIONS,
                'templates_dir': cls.TEMPLATES_DIR,
                'results_dir': cls.RESULTS_DIR,
                'logs_dir': cls.LOGS_DIR
            },
            'model_info': {
                'description': model_config.get('description', 'Unknown'),
                'max_input_tokens': model_config.get('max_input_tokens', 0),
                'cost_per_1k_input': model_config.get('cost_per_1k_input', 0),
                'cost_per_1k_output': model_config.get('cost_per_1k_output', 0)
            },
            'logging': {
                'level': cls.LOG_LEVEL,
                'format': cls.LOG_FORMAT
            }
        }
    
    @classmethod
    def create_directories(cls) -> Dict[str, bool]:
        """
        Create required directories if they don't exist
        
        Returns:
            Dictionary showing which directories were created
        """
        directories = {
            cls.RESULTS_DIR: False,
            cls.LOGS_DIR: False,
            cls.TEMPLATES_DIR: False
        }
        
        for directory in directories.keys():
            try:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    directories[directory] = True
                    logger.info(f"Created directory: {directory}")
                else:
                    logger.debug(f"Directory already exists: {directory}")
                    
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                directories[directory] = False
        
        return directories
    
    @classmethod
    def estimate_cost(cls, input_tokens: int, output_tokens: int, 
                     model_id: Optional[str] = None) -> Dict[str, float]:
        """
        Estimate cost for a grading operation
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_id: Model to use for cost calculation
            
        Returns:
            Cost breakdown dictionary
        """
        model_config = cls.get_model_config(model_id)
        
        input_cost = (input_tokens / 1000) * model_config['cost_per_1k_input']
        output_cost = (output_tokens / 1000) * model_config['cost_per_1k_output']
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost_usd': round(input_cost, 6),
            'output_cost_usd': round(output_cost, 6),
            'total_cost_usd': round(total_cost, 6),
            'model_used': model_id or cls.BEDROCK_MODEL_ID
        }
    
    @classmethod
    def get_assignment_structure_template(cls) -> Dict[str, List[str]]:
        """
        Get the expected S3 assignment structure
        
        Returns:
            Dictionary showing expected folder structure
        """
        return {
            'required_folders': cls.REQUIRED_FOLDERS.copy(),
            'optional_folders': cls.OPTIONAL_FOLDERS.copy(),
            'example_structure': [
                'assignment-name/',
                '├── question/',
                '│   └── case_study.pdf',
                '├── solution_keys/',
                '│   ├── solution_1.pdf',
                '│   └── solution_2.pdf',
                '└── student_responses/',
                '    ├── student_001.pdf',
                '    ├── student_002.pdf',
                '    └── student_003.pdf'
            ]
        }
    
    @classmethod
    def load_from_file(cls, config_file: str) -> bool:
        """
        Load configuration from a JSON file
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_file}")
                return False
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update class attributes from file
            for key, value in config_data.items():
                if hasattr(cls, key.upper()):
                    setattr(cls, key.upper(), value)
                    logger.debug(f"Updated config: {key.upper()} = {value}")
            
            logger.info(f"Configuration loaded from: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            return False
    
    @classmethod
    def save_to_file(cls, config_file: str) -> bool:
        """
        Save current configuration to a JSON file
        
        Args:
            config_file: Path to save configuration JSON file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_data = {
                'aws_region': cls.AWS_REGION,
                's3_bucket_name': cls.S3_BUCKET_NAME,
                'bedrock_model_id': cls.BEDROCK_MODEL_ID,
                'max_tokens': cls.MAX_TOKENS,
                'default_temperature': cls.DEFAULT_TEMPERATURE,
                'templates_dir': cls.TEMPLATES_DIR,
                'results_dir': cls.RESULTS_DIR,
                'logs_dir': cls.LOGS_DIR,
                'max_file_size_mb': cls.MAX_FILE_SIZE_MB,
                'log_level': cls.LOG_LEVEL
            }
            
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_file}: {e}")
            return False
    
    @classmethod
    def get_environment_info(cls) -> Dict[str, Any]:
        """
        Get information about the current environment
        
        Returns:
            Environment information dictionary
        """
        return {
            'environment_variables': {
                'AWS_REGION': os.getenv('AWS_REGION'),
                'S3_BUCKET_NAME': os.getenv('S3_BUCKET_NAME'),
                'BEDROCK_MODEL_ID': os.getenv('BEDROCK_MODEL_ID'),
                'MAX_TOKENS': os.getenv('MAX_TOKENS'),
                'LOG_LEVEL': os.getenv('LOG_LEVEL')
            },
            'current_working_directory': os.getcwd(),
            'config_file_locations': [
                '.env',
                'config.json',
                f'{os.path.expanduser("~")}/.llm_grader/config.json'
            ],
            'directory_status': {
                cls.RESULTS_DIR: Path(cls.RESULTS_DIR).exists(),
                cls.LOGS_DIR: Path(cls.LOGS_DIR).exists(),
                cls.TEMPLATES_DIR: Path(cls.TEMPLATES_DIR).exists()
            }
        }
    
    @classmethod
    def setup_logging(cls) -> bool:
        """
        Setup logging configuration based on config settings
        
        Returns:
            True if logging setup successfully
        """
        try:
            # Create logs directory
            cls.create_directories()
            
            # Configure logging
            import logging.config
            
            logging_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'standard': {
                        'format': cls.LOG_FORMAT
                    },
                },
                'handlers': {
                    'console': {
                        'level': cls.LOG_LEVEL,
                        'class': 'logging.StreamHandler',
                        'formatter': 'standard',
                    },
                    'file': {
                        'level': 'DEBUG',
                        'class': 'logging.FileHandler',
                        'filename': f'{cls.LOGS_DIR}/grader.log',
                        'formatter': 'standard',
                        'mode': 'a',
                    },
                },
                'loggers': {
                    '': {  # root logger
                        'handlers': ['console', 'file'],
                        'level': 'DEBUG',
                        'propagate': False
                    }
                }
            }
            
            logging.config.dictConfig(logging_config)
            logger.info(f"Logging configured: level={cls.LOG_LEVEL}, logs_dir={cls.LOGS_DIR}")
            
            return True
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            return False
    
    @classmethod
    def print_config(cls):
        """Print current configuration to console"""
        print("\n" + "="*60)
        print("LLM GRADER CONFIGURATION")
        print("="*60)
        
        summary = cls.get_config_summary()
        
        print(f"AWS Configuration:")
        print(f"  Region: {summary['aws']['region']}")
        print(f"  S3 Bucket: {summary['aws']['s3_bucket']}")
        print(f"  Bedrock Model: {summary['aws']['bedrock_model']}")
        
        print(f"\nGrading Settings:")
        print(f"  Max Tokens: {summary['grading']['max_tokens']}")
        print(f"  Temperature: {summary['grading']['temperature']}")
        print(f"  Top P: {summary['grading']['top_p']}")
        
        print(f"\nModel Information:")
        print(f"  Description: {summary['model_info']['description']}")
        print(f"  Max Input Tokens: {summary['model_info']['max_input_tokens']:,}")
        print(f"  Cost per 1K Input: ${summary['model_info']['cost_per_1k_input']}")
        print(f"  Cost per 1K Output: ${summary['model_info']['cost_per_1k_output']}")
        
        print(f"\nFile Configuration:")
        print(f"  Max File Size: {summary['files']['max_size_mb']} MB")
        print(f"  Supported Extensions: {summary['files']['supported_extensions']}")
        print(f"  Templates Directory: {summary['files']['templates_dir']}")
        print(f"  Results Directory: {summary['files']['results_dir']}")
        print(f"  Logs Directory: {summary['files']['logs_dir']}")
        
        print(f"\nLogging:")
        print(f"  Level: {summary['logging']['level']}")
        
        print("="*60)

# Import json module for file operations
import json

# Initialize logging when module is imported
Config.setup_logging()