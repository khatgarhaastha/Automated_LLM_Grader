import os
import json
from typing import Dict, List, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompt templates and generation using Jinja2
    
    This class handles loading, validating, and rendering Jinja2 templates
    for generating grading prompts with dynamic content.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """
        Initialize the prompt manager
        
        Args:
            templates_dir: Directory containing Jinja2 templates
        """
        self.templates_dir = templates_dir
        self.ensure_templates_dir()
        
        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Load default template variables
        self.default_vars = self.load_default_template_vars()
        
        logger.info(f"PromptManager initialized with templates dir: {self.templates_dir}")
        
        # Ensure default template exists
        self.create_default_template()
    
    def ensure_templates_dir(self):
        """Ensure templates directory exists"""
        templates_path = Path(self.templates_dir)
        if not templates_path.exists():
            templates_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created templates directory: {self.templates_dir}")
    
    def load_default_template_vars(self) -> Dict[str, str]:
        """Load default template variables for grading prompts"""
        return {
            # Headers and sections
            'system_prompt_header': 'SYSTEM PROMPT:',
            'case_study_header': 'CASE STUDY QUESTION:',
            'solution_keys_header': 'SOLUTION KEYS:',
            'examples_header': 'GRADING EXAMPLES:',
            'additional_context_header': 'ADDITIONAL CONTEXT:',
            'target_response_header': 'STUDENT RESPONSE TO GRADE:',
            
            # Labels and prefixes
            'solution_key_prefix': 'Solution Key',
            'example_prefix': 'Example',
            'student_response_label': 'Student Response',
            'grade_given_label': 'Grade Given',
            'feedback_label': 'Feedback',
            
            # Core grading instructions
            'grading_instructions': '''You are an expert academic grader. Your task is to evaluate student assignments based on provided case study questions, solution keys, and grading examples.''',
            
            # Grading criteria
            'grading_criteria': '''GRADING CRITERIA:
- Accuracy of analysis and conclusions
- Depth of understanding demonstrated
- Quality of reasoning and argumentation
- Use of relevant concepts and frameworks
- Clarity of communication
- Completeness of response''',
            
            # Grading scale
            'grading_scale': '''GRADING SCALE:
- A (90-100): Excellent - Comprehensive, insightful, well-reasoned
- B (80-89): Good - Solid understanding with minor gaps
- C (70-79): Average - Basic understanding with some issues
- D (60-69): Below Average - Significant gaps in understanding
- F (0-59): Failing - Major deficiencies''',
            
            # Output format
            'output_format': '''OUTPUT FORMAT:
Provide your evaluation in the following structure:
1. Grade: [Letter Grade] ([Numeric Score])
2. Strengths: [List key strengths]
3. Areas for Improvement: [List specific areas needing work]
4. Detailed Feedback: [Paragraph explaining the grade with specific examples]
5. Suggestions: [Actionable recommendations for improvement]''',
            
            # Final instructions
            'final_instructions': 'Please provide your evaluation following the output format specified above.'
        }
    
    def create_default_template(self) -> str:
        """
        Create and save the default grading template if it doesn't exist
        
        Returns:
            Path to the created template file
        """
        template_path = Path(self.templates_dir) / "grading_prompt.j2"
        
        if template_path.exists():
            logger.debug(f"Default template already exists: {template_path}")
            return str(template_path)
        
        # Default template content
        template_content = '''{# Grading Prompt Template #}
{# This is a Jinja2 template for generating comprehensive grading prompts #}

{# System Prompt Section #}
{{ grading_instructions }}

{{ grading_criteria }}

{{ grading_scale }}

{{ output_format }}

{# Case Study Section #}
{% if case_study_question %}
{{ case_study_header }}
{{ case_study_question }}

{% endif %}

{# Solution Keys Section #}
{% if solution_keys and solution_keys|length > 0 %}
{{ solution_keys_header }}
{% for solution in solution_keys %}
{{ solution_key_prefix }} {{ loop.index }}:
{{ solution.content }}

{% endfor %}
{% endif %}

{# Few-Shot Examples Section #}
{% if few_shot_examples and few_shot_examples|length > 0 %}
{{ examples_header }}
{% for example in few_shot_examples %}
{{ example_prefix }} {{ loop.index }} - {{ example.grade }} Grade:
{{ student_response_label }}: {{ example.response }}
{{ grade_given_label }}: {{ example.grade }}
{{ feedback_label }}: {{ example.feedback }}

{% endfor %}
{% endif %}

{# Additional Context Section #}
{% if additional_context %}
{{ additional_context_header }}
{{ additional_context }}

{% endif %}

{# Target Response Section #}
{{ target_response_header }}
{{ target_response }}

{# Final Instructions #}
{{ final_instructions }}'''
        
        # Save template
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            logger.info(f"Created default template: {template_path}")
            return str(template_path)
            
        except Exception as e:
            logger.error(f"Failed to create default template: {e}")
            raise
    
    def load_template(self, template_name: str = "grading_prompt.j2") -> Template:
        """
        Load a Jinja2 template
        
        Args:
            template_name: Name of the template file
            
        Returns:
            Loaded Jinja2 template object
        """
        try:
            template = self.env.get_template(template_name)
            logger.debug(f"Loaded template: {template_name}")
            return template
            
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            
            # If it's the default template, create it
            if template_name == "grading_prompt.j2":
                logger.info("Creating missing default template")
                self.create_default_template()
                return self.env.get_template(template_name)
            
            # List available templates for debugging
            available = self.list_templates()
            logger.error(f"Available templates: {available}")
            raise FileNotFoundError(f"Template '{template_name}' not found. Available: {available}")
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
            raise
    
    def generate_prompt(self, template_name: str = "grading_prompt.j2", **kwargs) -> str:
        """
        Generate a prompt using the specified template
        
        Args:
            template_name: Name of the template file
            **kwargs: Variables to pass to the template
            
        Returns:
            Generated prompt string
        """
        try:
            logger.debug(f"Generating prompt using template: {template_name}")
            
            # Load template
            template = self.load_template(template_name)
            
            # Merge default variables with provided kwargs
            template_vars = {**self.default_vars, **kwargs}
            
            # Log template variables for debugging
            logger.debug(f"Template variables: {list(template_vars.keys())}")
            
            # Render template
            prompt = template.render(**template_vars)
            
            # Basic validation
            if not prompt or len(prompt.strip()) < 100:
                logger.warning(f"Generated prompt seems too short ({len(prompt)} chars)")
            
            logger.info(f"Generated prompt using template: {template_name} ({len(prompt)} characters)")
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error generating prompt with template '{template_name}': {e}")
            raise
    
    def list_templates(self) -> List[str]:
        """
        List all available templates
        
        Returns:
            List of template filenames
        """
        try:
            templates_path = Path(self.templates_dir)
            
            if not templates_path.exists():
                logger.warning(f"Templates directory does not exist: {self.templates_dir}")
                return []
            
            # Find all .j2 files
            templates = []
            for file_path in templates_path.glob("*.j2"):
                templates.append(file_path.name)
            
            logger.debug(f"Found {len(templates)} templates")
            return sorted(templates)
            
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        Validate a template for syntax errors and basic functionality
        
        Args:
            template_name: Name of the template to validate
            
        Returns:
            Validation results dictionary
        """
        try:
            logger.info(f"Validating template: {template_name}")
            
            # Try to load the template
            template = self.load_template(template_name)
            
            # Test rendering with minimal data
            test_data = {
                'target_response': 'test response',
                'case_study_question': 'test question',
                'solution_keys': [{'content': 'test solution'}],
                'few_shot_examples': [{
                    'grade': 'A (95)',
                    'response': 'test excellent response',
                    'feedback': 'test excellent feedback'
                }]
            }
            
            # Render with test data
            rendered = template.render(**{**self.default_vars, **test_data})
            
            # Basic checks
            validation_result = {
                'valid': True,
                'template_name': template_name,
                'message': 'Template validation successful',
                'rendered_length': len(rendered),
                'contains_required_sections': {
                    'grading_instructions': 'grading_instructions' in rendered.lower(),
                    'case_study': 'case study' in rendered.lower(),
                    'solution_keys': 'solution key' in rendered.lower(),
                    'target_response': 'test response' in rendered,
                    'final_instructions': 'evaluation' in rendered.lower()
                }
            }
            
            # Check if all required sections are present
            missing_sections = [section for section, present in 
                              validation_result['contains_required_sections'].items() 
                              if not present]
            
            if missing_sections:
                validation_result['warnings'] = [f"Missing sections: {', '.join(missing_sections)}"]
            
            logger.info(f"✅ Template validation successful: {template_name}")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Template validation failed for '{template_name}': {e}")
            return {
                'valid': False,
                'template_name': template_name,
                'error': str(e),
                'message': f'Template validation failed: {str(e)}'
            }
    
    def preview_template(self, template_name: str = "grading_prompt.j2", 
                        sample_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Preview a template with sample data
        
        Args:
            template_name: Name of the template to preview
            sample_data: Optional sample data (uses defaults if not provided)
            
        Returns:
            Rendered template preview
        """
        try:
            logger.info(f"Generating template preview: {template_name}")
            
            # Default sample data if none provided
            if sample_data is None:
                sample_data = {
                    'case_study_question': '''[SAMPLE CASE STUDY]
ABC Corporation is facing declining market share in the smartphone industry. 
The company needs to decide whether to invest in 5G technology development 
or focus on improving their existing 4G product line. Consider the financial 
implications, market trends, and competitive landscape in your analysis.''',
                    
                    'solution_keys': [
                        {
                            'content': '''[SAMPLE SOLUTION KEY 1]
Key considerations should include:
1. Market analysis of 5G adoption rates
2. Financial projections for both options
3. Competitive positioning
4. Risk assessment
5. Recommendation with clear justification'''
                        },
                        {
                            'content': '''[SAMPLE SOLUTION KEY 2]
Strong responses will demonstrate:
- Understanding of technology lifecycle
- Financial modeling capabilities  
- Strategic thinking
- Clear communication of recommendations'''
                        }
                    ],
                    
                    'few_shot_examples': [
                        {
                            'grade': 'A (95)',
                            'response': '''[SAMPLE EXCELLENT RESPONSE]
ABC Corporation should invest in 5G technology development for the following reasons:
1. Market Analysis: 5G adoption is accelerating globally...
2. Financial Projections: Initial investment of $500M will yield...
3. Competitive Advantage: Early entry into 5G market will...
[Response continues with detailed analysis]''',
                            'feedback': '''Excellent analysis demonstrating comprehensive understanding 
of the strategic decision. The response includes thorough market analysis, 
detailed financial projections, and clear strategic reasoning. The recommendation 
is well-supported with evidence and shows sophisticated business thinking.'''
                        },
                        {
                            'grade': 'B (83)',
                            'response': '''[SAMPLE GOOD RESPONSE]
I think ABC should focus on 5G because it's the future. The company
needs to stay competitive and 5G is important for smartphones...
[Response shows basic understanding but lacks depth]''',
                            'feedback': '''Good understanding of the basic strategic choice, but 
the analysis lacks depth in financial modeling and market analysis. 
The recommendation is sound but needs stronger supporting evidence 
and more detailed consideration of risks and alternatives.'''
                        },
                        {
                            'grade': 'C (72)',
                            'response': '''[SAMPLE AVERAGE RESPONSE]
ABC Corporation has two options. 5G is new technology and 4G is old.
I think they should do 5G because everyone is doing it...
[Response shows minimal analysis]''',
                            'feedback': '''Basic understanding of the decision but minimal analytical 
depth. The response lacks financial analysis, market research, and 
strategic reasoning. Recommendation appears to be based on general 
assumptions rather than thorough analysis.'''
                        }
                    ],
                    
                    'target_response': '''[SAMPLE STUDENT RESPONSE TO BE GRADED]
ABC Corporation should invest in 5G technology development. The smartphone
market is evolving rapidly and companies that don't adapt will lose market share.
5G offers faster speeds and better connectivity which customers want.

Financial Analysis:
- 5G development requires significant upfront investment
- Market research shows 5G adoption growing at 25% annually
- Competitors are already investing in 5G capabilities

Recommendation: Invest in 5G development while maintaining core 4G products
for existing customer base. This balanced approach minimizes risk while 
positioning the company for future growth.''',
                    
                    'additional_context': '''[SAMPLE ADDITIONAL CONTEXT]
This is a midterm exam question worth 25% of the final grade. 
Students have had 3 weeks to prepare and should demonstrate 
understanding of strategic analysis frameworks covered in class.'''
                }
            
            # Generate preview
            preview = self.generate_prompt(template_name, **sample_data)
            
            logger.info(f"Generated template preview for: {template_name} ({len(preview)} characters)")
            return preview
            
        except Exception as e:
            logger.error(f"Error generating template preview: {e}")
            return f"Error generating preview for '{template_name}': {str(e)}"
    
    def save_custom_template(self, template_name: str, content: str) -> str:
        """
        Save a custom template
        
        Args:
            template_name: Name for the new template (should end with .j2)
            content: Template content
            
        Returns:
            Path to the saved template
        """
        try:
            # Ensure .j2 extension
            if not template_name.endswith('.j2'):
                template_name += '.j2'
            
            template_path = Path(self.templates_dir) / template_name
            
            # Create backup if template already exists
            if template_path.exists():
                backup_path = template_path.with_suffix(f'.j2.backup.{int(time.time())}')
                template_path.rename(backup_path)
                logger.info(f"Backed up existing template to: {backup_path}")
            
            # Save new template
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Saved custom template: {template_path}")
            
            # Validate the new template
            validation = self.validate_template(template_name)
            if not validation['valid']:
                logger.warning(f"Saved template has validation issues: {validation['error']}")
            
            return str(template_path)
            
        except Exception as e:
            logger.error(f"Error saving custom template: {e}")
            raise
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get information about a specific template
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template information dictionary
        """
        try:
            template_path = Path(self.templates_dir) / template_name
            
            if not template_path.exists():
                return {
                    'exists': False,
                    'template_name': template_name,
                    'error': 'Template file not found'
                }
            
            # Get file stats
            stat = template_path.stat()
            
            # Read template content
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Analyze template content
            info = {
                'exists': True,
                'template_name': template_name,
                'file_path': str(template_path),
                'file_size': stat.st_size,
                'last_modified': stat.st_mtime,
                'line_count': content.count('\n') + 1,
                'character_count': len(content),
                'contains_loops': '{%' in content and 'for' in content,
                'contains_conditionals': '{%' in content and 'if' in content,
                'variable_count': content.count('{{'),
                'validation': self.validate_template(template_name)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting template info: {e}")
            return {
                'exists': False,
                'template_name': template_name,
                'error': str(e)
            }
    
    def get_template_vars(self) -> Dict[str, str]:
        """
        Get current default template variables
        
        Returns:
            Dictionary of template variables
        """
        return self.default_vars.copy()
    
    def update_template_vars(self, new_vars: Dict[str, str]):
        """
        Update default template variables
        
        Args:
            new_vars: Dictionary of new/updated variables
        """
        original_count = len(self.default_vars)
        self.default_vars.update(new_vars)
        
        logger.info(f"Updated template variables: {len(new_vars)} new/changed, "
                   f"total: {len(self.default_vars)} (was {original_count})")
    
    def create_template_from_example(self, template_name: str, 
                                   base_template: str = "grading_prompt.j2") -> str:
        """
        Create a new template based on an existing one
        
        Args:
            template_name: Name for the new template
            base_template: Name of the template to copy from
            
        Returns:
            Path to the new template
        """
        try:
            # Load base template content
            base_path = Path(self.templates_dir) / base_template
            
            if not base_path.exists():
                raise FileNotFoundError(f"Base template not found: {base_template}")
            
            with open(base_path, 'r', encoding='utf-8') as f:
                base_content = f.read()
            
            # Add header comment
            new_content = (
    f"{{# Custom template based on {base_template} #}}\n"
    f"{{# Created: {datetime.now().isoformat()} #}}\n"
    f"{{# Modify this template to customize your grading prompts #}}\n\n"
    f"{base_content}"
)
            
            # Save new template
            new_template_path = self.save_custom_template(template_name, new_content)
            
            logger.info(f"Created new template '{template_name}' based on '{base_template}'")
            
            return new_template_path
            
        except Exception as e:
            logger.error(f"Error creating template from example: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of the prompt manager"""
        template_count = len(self.list_templates())
        return f"PromptManager(templates_dir='{self.templates_dir}', templates={template_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"PromptManager(templates_dir='{self.templates_dir}', template_vars={len(self.default_vars)})"