#!/usr/bin/env python3
"""
Enhanced LLM Grader - Main Execution Script
This script provides multiple modes for running the LLM grader system
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Src'))

from grader import LLMGrader
from bedrock_client import BedrockClient
from s3_client import S3Client
from prompt_manager import PromptManager
from config import Config

def setup_logging():
    """Setup comprehensive logging configuration"""
    import logging
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(
        logs_dir / f'grader_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def load_few_shot_examples(examples_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load few-shot examples from file or return defaults
    
    Args:
        examples_file: Path to JSON file containing examples
        
    Returns:
        List of few-shot examples
    """
    # Try to load from specified file
    if examples_file and os.path.exists(examples_file):
        try:
            with open(examples_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            # Validate examples structure
            required_keys = ['grade', 'response', 'feedback']
            for i, example in enumerate(examples):
                if not all(key in example for key in required_keys):
                    raise ValueError(f"Example {i} missing required keys: {required_keys}")
            
            print(f"✓ Loaded {len(examples)} few-shot examples from {examples_file}")
            return examples
            
        except Exception as e:
            print(f"⚠️ Error loading examples file {examples_file}: {e}")
            print("Using default examples instead")
    
    # Try default locations
    default_paths = [
        'TestData/sample_input.json',
        'TestData/few_shot_examples.json',
        'examples.json'
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                print(f"✓ Loaded {len(examples)} examples from {path}")
                return examples
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
    
    # Return default examples from config
    print("📝 Using default few-shot examples from config")
    return Config.get_few_shot_examples()

def load_target_response(response_input: str) -> str:
    """
    Load target response from file or return as-is
    
    Args:
        response_input: Either file path or direct response text
        
    Returns:
        Target response text
    """
    # Check if it's a file path
    if os.path.exists(response_input):
        try:
            with open(response_input, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            print(f"✓ Loaded target response from file: {response_input}")
            return content
        except Exception as e:
            print(f"⚠️ Error loading response file {response_input}: {e}")
            return response_input
    
    # Return as direct text
    return response_input

def save_results(result: Dict[str, Any], output_dir: str = "results") -> Dict[str, str]:
    """
    Save grading results with detailed metadata
    
    Args:
        result: Grading result dictionary
        output_dir: Output directory for results
        
    Returns:
        Dictionary with saved file paths
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    assignment_name = result.get('assignment_name', 'unknown')
    
    # Save complete result as JSON
    json_filename = f"{assignment_name}_grading_{timestamp}.json"
    json_filepath = output_path / json_filename
    
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    
    # Save evaluation text for easy reading
    txt_filename = f"{assignment_name}_evaluation_{timestamp}.txt"
    txt_filepath = output_path / txt_filename
    
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(f"Assignment: {assignment_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Template Used: {result.get('template_used', 'unknown')}\n")
        f.write(f"Success: {result.get('success', False)}\n")
        f.write("=" * 60 + "\n\n")
        
        if result.get('success'):
            f.write("GRADING EVALUATION:\n")
            f.write("-" * 20 + "\n")
            f.write(result.get('evaluation', 'No evaluation available'))
            f.write("\n\n")
            
            # Add metadata
            metadata = result.get('metadata', {})
            f.write("METADATA:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Materials loaded: {metadata.get('materials_loaded', {})}\n")
            f.write(f"Model info: {metadata.get('model_info', {})}\n")
            f.write(f"Prompt length: {metadata.get('prompt_length', 'unknown')} tokens\n")
        else:
            f.write("ERROR:\n")
            f.write("-" * 20 + "\n")
            f.write(result.get('error', 'Unknown error'))
    
    # Save prompt for debugging (if available)
    if 'prompt' in result:
        prompt_filename = f"{assignment_name}_prompt_{timestamp}.txt"
        prompt_filepath = output_path / prompt_filename
        
        with open(prompt_filepath, 'w', encoding='utf-8') as f:
            f.write(result['prompt'])
    
    print(f"📁 Results saved to: {json_filepath}")
    print(f"📄 Evaluation saved to: {txt_filepath}")
    
    return {
        'json_file': str(json_filepath),
        'txt_file': str(txt_filepath),
        'prompt_file': str(prompt_filepath) if 'prompt' in result else None
    }

def display_system_status(grader: LLMGrader):
    """Display comprehensive system status"""
    print("\n" + "=" * 60)
    print("SYSTEM STATUS")
    print("=" * 60)
    
    status = grader.get_system_status()
    
    # Overall health
    overall_health = status['connections']['overall_status']
    health_emoji = "✅" if overall_health == 'healthy' else "❌"
    print(f"{health_emoji} Overall Status: {overall_health.upper()}")
    
    # Component status
    print(f"\n📊 COMPONENT STATUS:")
    bedrock_status = status['connections']['bedrock']
    s3_status = status['connections']['s3']
    
    print(f"  🤖 Bedrock: {'✅' if bedrock_status['success'] else '❌'}")
    if not bedrock_status['success']:
        print(f"      Error: {bedrock_status.get('error', 'Unknown error')}")
    
    print(f"  📁 S3: {'✅' if s3_status['success'] else '❌'}")
    if not s3_status['success']:
        print(f"      Error: {s3_status.get('error', 'Unknown error')}")
    
    # Configuration
    print(f"\n⚙️ CONFIGURATION:")
    print(f"  Region: {status['config']['region']}")
    print(f"  S3 Bucket: {status['s3_bucket']}")
    print(f"  Model: {status['bedrock_model']['model_id']}")
    print(f"  Max Tokens: {status['config']['max_tokens']}")
    
    # Templates
    print(f"\n📋 TEMPLATES:")
    templates = status['available_templates']
    for template in templates:
        print(f"  📄 {template}")
    
    print("=" * 60)

def interactive_mode(grader: LLMGrader):
    """Interactive mode for testing and configuration"""
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 40)
    
    while True:
        print("\nSelect an option:")
        print("1. 📊 System Status")
        print("2. 📋 List Templates")
        print("3. 👀 Preview Template")
        print("4. 📁 Assignment Info")
        print("5. 🔍 Preview Grading Prompt")
        print("6. 🎯 Grade Assignment")
        print("7. 🔄 Batch Grade")
        print("8. 🧪 Test Connection")
        print("9. 🚪 Exit")
        
        try:
            choice = input("\nEnter your choice (1-9): ").strip()
            
            if choice == '1':
                display_system_status(grader)
            
            elif choice == '2':
                templates = grader.prompt_manager.list_templates()
                print(f"\n📋 Available Templates ({len(templates)}):")
                for i, template in enumerate(templates, 1):
                    print(f"  {i}. {template}")
            
            elif choice == '3':
                template_name = input("Enter template name (or press Enter for default): ").strip()
                if not template_name:
                    template_name = "grading_prompt.j2"
                
                try:
                    preview = grader.prompt_manager.preview_template(template_name)
                    print(f"\n📄 Template Preview: {template_name}")
                    print("-" * 50)
                    print(preview)
                except Exception as e:
                    print(f"❌ Error previewing template: {e}")
            
            elif choice == '4':
                assignment_name = input("Enter assignment name: ").strip()
                if assignment_name:
                    info = grader.get_assignment_info(assignment_name)
                    print(f"\n📁 Assignment Info: {assignment_name}")
                    print("-" * 40)
                    print(json.dumps(info, indent=2, default=str))
            
            elif choice == '5':
                assignment_name = input("Enter assignment name: ").strip()
                if assignment_name:
                    examples = load_few_shot_examples()
                    prompt = grader.preview_grading_prompt(assignment_name, examples)
                    print(f"\n🔍 Grading Prompt Preview: {assignment_name}")
                    print("-" * 60)
                    print(prompt)
            
            elif choice == '6':
                assignment_name = input("Enter assignment name: ").strip()
                if not assignment_name:
                    print("❌ Assignment name is required")
                    continue
                
                response_input = input("Enter target response (text or file path): ").strip()
                if not response_input:
                    print("❌ Target response is required")
                    continue
                
                target_response = load_target_response(response_input)
                examples = load_few_shot_examples()
                
                print(f"\n🎯 Grading Assignment: {assignment_name}")
                print("⏳ Processing...")
                
                result = grader.grade_assignment(
                    assignment_name=assignment_name,
                    target_response=target_response,
                    few_shot_examples=examples
                )
                
                if result['success']:
                    print(f"\n✅ GRADING COMPLETED")
                    print("=" * 50)
                    print(result['evaluation'])
                    save_results(result)
                else:
                    print(f"❌ Grading failed: {result['error']}")
            
            elif choice == '7':
                print("🔄 Batch grading not implemented in interactive mode")
                print("Use command line: python main_grader.py --batch assignments.json")
            
            elif choice == '8':
                print("🧪 Testing connections...")
                display_system_status(grader)
            
            elif choice == '9':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def standard_grading_mode(grader: LLMGrader, assignment_name: str, 
                         target_response: str, examples_file: Optional[str] = None,
                         template_name: str = "grading_prompt.j2") -> Dict[str, Any]:
    """
    Standard grading mode for single assignment
    
    Args:
        grader: LLMGrader instance
        assignment_name: Name of assignment to grade
        target_response: Student response to grade
        examples_file: Path to few-shot examples file
        template_name: Template to use for grading
        
    Returns:
        Grading result dictionary
    """
    print(f"\n🎯 GRADING ASSIGNMENT: {assignment_name}")
    print("=" * 50)
    
    # Load few-shot examples
    few_shot_examples = load_few_shot_examples(examples_file)
    
    # Load target response
    target_text = load_target_response(target_response)
    
    print(f"📋 Template: {template_name}")
    print(f"📝 Few-shot examples: {len(few_shot_examples)}")
    print(f"📄 Target response length: {len(target_text)} characters")
    print("⏳ Processing...")
    
    # Grade the assignment
    result = grader.grade_assignment(
        assignment_name=assignment_name,
        target_response=target_text,
        few_shot_examples=few_shot_examples,
        template_name=template_name
    )
    
    return result

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Grader - Automated assignment grading using Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_grader.py --interactive
  python main_grader.py --assignment case-study-1 --response "student response text"
  python main_grader.py --assignment case-study-1 --response-file response.txt
  python main_grader.py --status
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    parser.add_argument('--status', action='store_true',
                       help='Show system status and exit')
    
    parser.add_argument('--assignment', type=str,
                       help='Assignment name to grade')
    
    parser.add_argument('--response', type=str,
                       help='Target response text to grade')
    
    parser.add_argument('--response-file', type=str,
                       help='File containing target response to grade')
    
    parser.add_argument('--examples', type=str,
                       help='JSON file containing few-shot examples')
    
    parser.add_argument('--template', type=str, default='grading_prompt.j2',
                       help='Template to use for grading')
    
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display header
    print("🚀 LLM GRADER")
    print("=" * 40)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Validate configuration
        print("⚙️ Validating configuration...")
        config_validation = Config.validate_config()
        
        if not config_validation['is_valid']:
            print("❌ Configuration errors:")
            for error in config_validation['errors']:
                print(f"  • {error}")
            return 1
        
        if config_validation['warnings']:
            print("⚠️ Configuration warnings:")
            for warning in config_validation['warnings']:
                print(f"  • {warning}")
        
        # Initialize system components
        print("🔧 Initializing components...")
        bedrock_client = BedrockClient()
        s3_client = S3Client()
        prompt_manager = PromptManager()
        
        # Ensure default template exists
        prompt_manager.create_default_template()
        
        # Initialize grader
        grader = LLMGrader(bedrock_client, s3_client, prompt_manager)
        
        # Handle different modes
        if args.status:
            display_system_status(grader)
            return 0
        
        if args.interactive:
            interactive_mode(grader)
            return 0
        
        # Standard grading mode
        if args.assignment:
            # Determine target response
            if args.response_file:
                target_response = args.response_file
            elif args.response:
                target_response = args.response
            else:
                print("❌ Error: Either --response or --response-file must be provided")
                return 1
            
            # Grade the assignment
            result = standard_grading_mode(
                grader=grader,
                assignment_name=args.assignment,
                target_response=target_response,
                examples_file=args.examples,
                template_name=args.template
            )
            
            # Display and save results
            if result['success']:
                print(f"\n✅ GRADING COMPLETED")
                print("=" * 50)
                print(result['evaluation'])
                
                # Display metadata
                metadata = result.get('metadata', {})
                print(f"\n📊 METADATA:")
                print(f"  Model: {metadata.get('model_info', {}).get('model_id', 'Unknown')}")
                print(f"  Tokens: {metadata.get('model_info', {}).get('input_tokens', 0)} + {metadata.get('model_info', {}).get('output_tokens', 0)}")
                print(f"  Materials: {metadata.get('materials_loaded', {})}")
                
                # Save results
                save_results(result, args.output)
                
            else:
                print(f"❌ Grading failed: {result['error']}")
                return 1
        
        else:
            # No specific mode selected, show help
            print("ℹ️ No mode selected. Use --help for options or --interactive for interactive mode.")
            parser.print_help()
            return 0
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Operation cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        print(f"\n⏰ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)