import boto3
import PyPDF2
import pdfplumber
from io import BytesIO
import logging
import os
import mimetypes
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError, BotoCoreError
from config import Config

logger = logging.getLogger(__name__)


class S3Client:
    """
    Simplified S3 client for assignment materials management
    
    This client handles S3 operations and PDF text extraction for the LLM grader
    with simple validation and clean error handling.
    """
    
    def __init__(self, bucket_name: Optional[str] = None, region_name: Optional[str] = None):
        """
        Initialize S3 client
        
        Args:
            bucket_name: S3 bucket name (defaults to Config.S3_BUCKET_NAME)
            region_name: AWS region (defaults to Config.AWS_REGION)
        """
        self.bucket_name = bucket_name or Config.S3_BUCKET_NAME
        self.region_name = region_name or Config.AWS_REGION
        
        # Initialize boto3 client
        try:
            self.client = boto3.client('s3', region_name=self.region_name)
            logger.info(f"S3 client initialized: {self.region_name}, bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise ConnectionError(f"Cannot connect to S3: {e}")
    
    def test_bucket_access(self) -> Dict[str, Any]:
        """
        Test access to the S3 bucket
        
        Returns:
            Dict with test results
        """
        try:
            logger.info(f"Testing S3 bucket access: {self.bucket_name}")
            
            # Test bucket access
            response = self.client.head_bucket(Bucket=self.bucket_name)
            
            # Test list permissions
            try:
                list_response = self.client.list_objects_v2(
                    Bucket=self.bucket_name,
                    MaxKeys=1
                )
                can_list = True
            except Exception as e:
                logger.warning(f"Cannot list objects: {e}")
                can_list = False
            
            logger.info(f"S3 bucket access test successful: {self.bucket_name}")
            
            return {
                'success': True,
                'bucket_name': self.bucket_name,
                'region': self.region_name,
                'can_list_objects': can_list,
                'message': 'Bucket access successful'
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            
            logger.error(f"S3 bucket access test failed: {error_code} - {error_msg}")
            
            return {
                'success': False,
                'bucket_name': self.bucket_name,
                'region': self.region_name,
                'error': f"AWS Error ({error_code}): {error_msg}",
                'error_code': error_code
            }
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            return {
                'success': False,
                'bucket_name': self.bucket_name,
                'error': 'AWS credentials not found or invalid'
            }
            
        except Exception as e:
            logger.error(f"S3 bucket access test error: {e}")
            return {
                'success': False,
                'bucket_name': self.bucket_name,
                'error': str(e)
            }
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with given prefix
        
        Args:
            prefix: Object key prefix to filter by
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object information dictionaries
        """
        try:
            objects = []
            continuation_token = None
            total_fetched = 0
            
            while total_fetched < max_keys:
                list_params = {
                    'Bucket': self.bucket_name,
                    'Prefix': prefix,
                    'MaxKeys': min(max_keys - total_fetched, 1000)
                }
                
                if continuation_token:
                    list_params['ContinuationToken'] = continuation_token
                
                response = self.client.list_objects_v2(**list_params)
                
                for obj in response.get('Contents', []):
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
                    total_fetched += 1
                    
                    if total_fetched >= max_keys:
                        break
                
                if not response.get('IsTruncated', False):
                    break
                    
                continuation_token = response.get('NextContinuationToken')
            
            logger.info(f"Listed {len(objects)} objects with prefix: {prefix}")
            return objects
            
        except Exception as e:
            logger.error(f"Error listing S3 objects with prefix '{prefix}': {e}")
            return []
    
    def list_pdf_files(self, prefix: str = "") -> List[str]:
        """
        List only PDF files in S3 bucket with given prefix
        
        Args:
            prefix: Object key prefix to filter by
            
        Returns:
            List of PDF file keys
        """
        try:
            all_objects = self.list_objects(prefix)
            pdf_files = [obj['key'] for obj in all_objects 
                        if obj['key'].lower().endswith('.pdf')]
            
            logger.info(f"Found {len(pdf_files)} PDF files with prefix: {prefix}")
            return pdf_files
            
        except Exception as e:
            logger.error(f"Error listing PDF files: {e}")
            return []
    
    def extract_text_from_pdf(self, key: str, method: str = "auto") -> Dict[str, Any]:
        """
        Extract text from PDF stored in S3
        
        Args:
            key: S3 object key
            method: Extraction method ('auto', 'pdfplumber', 'pypdf2')
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Extracting text from: {key}")
        
        try:
            # Get PDF from S3
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            pdf_content = response['Body'].read()
            pdf_size = len(pdf_content)
            
            logger.info(f"Downloaded PDF: {key} ({pdf_size:,} bytes)")
            
            # Extract text with fallback
            extraction_result = self._extract_text_with_fallback(pdf_content, method)
            
            if extraction_result['success']:
                text = extraction_result['text']
                char_count = len(text)
                word_count = len(text.split()) if text else 0
                
                logger.info(f"Successfully extracted {char_count:,} characters from {key}")
                
                return {
                    'success': True,
                    'text': text,
                    'key': key,
                    'method_used': extraction_result['method_used'],
                    'character_count': char_count,
                    'word_count': word_count
                }
            else:
                logger.error(f"Failed to extract text from {key}: {extraction_result['error']}")
                
                return {
                    'success': False,
                    'text': '',
                    'key': key,
                    'error': extraction_result['error']
                }
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            
            logger.error(f"S3 error extracting text from {key}: {error_code} - {error_msg}")
            
            return {
                'success': False,
                'text': '',
                'key': key,
                'error': f"S3 Error ({error_code}): {error_msg}"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error extracting text from {key}: {e}")
            
            return {
                'success': False,
                'text': '',
                'key': key,
                'error': str(e)
            }
    
    def _extract_text_with_fallback(self, pdf_content: bytes, 
                                   preferred_method: str = "auto") -> Dict[str, Any]:
        """
        Extract text using multiple methods with intelligent fallback
        
        Args:
            pdf_content: PDF file content as bytes
            preferred_method: Preferred extraction method
            
        Returns:
            Dictionary with extraction results
        """
        methods_to_try = []
        
        if preferred_method == "auto":
            methods_to_try = ["pdfplumber", "pypdf2"]
        elif preferred_method == "pdfplumber":
            methods_to_try = ["pdfplumber", "pypdf2"]
        elif preferred_method == "pypdf2":
            methods_to_try = ["pypdf2", "pdfplumber"]
        else:
            methods_to_try = ["pdfplumber", "pypdf2"]
        
        for method in methods_to_try:
            try:
                if method == "pdfplumber":
                    text = self._extract_with_pdfplumber(pdf_content)
                elif method == "pypdf2":
                    text = self._extract_with_pypdf2(pdf_content)
                else:
                    continue
                
                # Check if extraction was successful
                if text and len(text.strip()) > 10:  # Minimum viable text
                    return {
                        'success': True,
                        'text': text.strip(),
                        'method_used': method
                    }
                    
            except Exception as e:
                logger.warning(f"Text extraction method {method} failed: {e}")
                continue
        
        # All methods failed
        return {
            'success': False,
            'text': '',
            'method_used': None,
            'error': f"All extraction methods failed"
        }
    
    def _extract_with_pdfplumber(self, pdf_content: bytes) -> str:
        """Extract text using pdfplumber"""
        text = ""
        
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return text
    
    def _extract_with_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2"""
        text = ""
        
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        return text
    
    def simple_assignment_check(self, assignment_name: str) -> Dict[str, Any]:
        """
        Simple check - verify basic files exist for grading
        No complex scoring, just practical yes/no answers
        
        Args:
            assignment_name: Name of the assignment to check
            
        Returns:
            Dictionary with simple validation results
        """
        try:
            logger.info(f"Checking assignment readiness: {assignment_name}")
            
            # Check each required folder
            question_files = self.list_pdf_files(f"{assignment_name}/question/")
            solution_files = self.list_pdf_files(f"{assignment_name}/solution_keys/")
            response_files = self.list_pdf_files(f"{assignment_name}/student_responses/")
            
            # Simple status
            result = {
                'assignment_name': assignment_name,
                'ready_for_grading': False,
                'has_question': len(question_files) > 0,
                'has_solutions': len(solution_files) > 0,
                'has_responses': len(response_files) > 0,
                'question_file': question_files[0] if question_files else None,
                'solution_count': len(solution_files),
                'response_count': len(response_files),
                'missing': []
            }
            
            # Check what's missing
            if not result['has_question']:
                result['missing'].append('question PDF')
            if not result['has_solutions']:
                result['missing'].append('solution key PDFs')
            # Note: student responses are optional for grading new assignments
            
            # Ready if we have question and at least one solution
            result['ready_for_grading'] = result['has_question'] and result['has_solutions']
            
            if result['ready_for_grading']:
                logger.info(f"✅ {assignment_name} is ready for grading")
                logger.info(f"   Question: ✅  Solutions: {result['solution_count']}  "
                           f"Examples: {result['response_count']}")
            else:
                logger.warning(f"⚠️ {assignment_name} is missing: {', '.join(result['missing'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking assignment: {e}")
            return {
                'assignment_name': assignment_name,
                'ready_for_grading': False,
                'error': str(e)
            }
    
    def get_assignment_structure(self, assignment_name: str) -> Dict[str, Any]:
        """
        Get basic structure information about an assignment
        
        Args:
            assignment_name: Name of the assignment folder
            
        Returns:
            Dictionary with assignment structure info
        """
        try:
            logger.info(f"Getting assignment structure: {assignment_name}")
            
            structure = {
                'assignment_name': assignment_name,
                'question': {
                    'files': [],
                    'file_count': 0
                },
                'solution_keys': {
                    'files': [],
                    'file_count': 0
                },
                'student_responses': {
                    'files': [],
                    'file_count': 0
                }
            }
            
            # Check each folder
            folders = {
                'question': f"{assignment_name}/question/",
                'solution_keys': f"{assignment_name}/solution_keys/",
                'student_responses': f"{assignment_name}/student_responses/"
            }
            
            for folder_type, prefix in folders.items():
                pdf_files = self.list_pdf_files(prefix)
                structure[folder_type]['files'] = pdf_files
                structure[folder_type]['file_count'] = len(pdf_files)
            
            logger.info(f"Structure loaded: Q:{structure['question']['file_count']} "
                       f"SK:{structure['solution_keys']['file_count']} "
                       f"SR:{structure['student_responses']['file_count']}")
            
            return structure
            
        except Exception as e:
            logger.error(f"Error getting assignment structure: {e}")
            return {
                'assignment_name': assignment_name,
                'error': str(e)
            }
    
    def load_assignment_materials(self, assignment_name: str) -> Dict[str, Any]:
        """
        Load all materials for an assignment with simple validation
        
        Args:
            assignment_name: Name of the assignment folder
            
        Returns:
            Dictionary with assignment materials and metadata
        """
        try:
            logger.info(f"Loading assignment materials: {assignment_name}")
            
            # First, do a simple check
            check = self.simple_assignment_check(assignment_name)
            
            if not check['ready_for_grading']:
                return {
                    'assignment_name': assignment_name,
                    'error': f"Assignment not ready for grading: missing {', '.join(check['missing'])}",
                    'check_result': check
                }
            
            materials = {
                'assignment_name': assignment_name,
                'question': '',
                'solution_keys': [],
                'student_responses': [],
                'metadata': {
                    'question_loaded': False,
                    'solution_keys_count': 0,
                    'student_responses_count': 0,
                    'extraction_errors': []
                }
            }
            
            # Load question
            if check['question_file']:
                question_result = self.extract_text_from_pdf(check['question_file'])
                if question_result['success']:
                    materials['question'] = question_result['text']
                    materials['metadata']['question_loaded'] = True
                    logger.info(f"✅ Question loaded from: {check['question_file']}")
                else:
                    materials['metadata']['extraction_errors'].append({
                        'file': check['question_file'],
                        'error': question_result['error']
                    })
                    logger.error(f"❌ Failed to extract question from {check['question_file']}")
            
            # Load solution keys
            solution_files = self.list_pdf_files(f"{assignment_name}/solution_keys/")
            for file in solution_files:
                result = self.extract_text_from_pdf(file)
                if result['success']:
                    materials['solution_keys'].append({
                        'filename': Path(file).name,
                        'content': result['text'],
                        'character_count': result['character_count'],
                        'word_count': result['word_count']
                    })
                    logger.info(f"✅ Solution key loaded: {Path(file).name}")
                else:
                    materials['metadata']['extraction_errors'].append({
                        'file': file,
                        'error': result['error']
                    })
                    logger.error(f"❌ Failed to extract solution key: {Path(file).name}")
            
            materials['metadata']['solution_keys_count'] = len(materials['solution_keys'])
            
            # Load student responses (optional)
            response_files = self.list_pdf_files(f"{assignment_name}/student_responses/")
            for file in response_files:
                result = self.extract_text_from_pdf(file)
                if result['success']:
                    materials['student_responses'].append({
                        'filename': Path(file).name,
                        'content': result['text'],
                        'character_count': result['character_count'],
                        'word_count': result['word_count']
                    })
                    logger.info(f"✅ Student response loaded: {Path(file).name}")
                else:
                    materials['metadata']['extraction_errors'].append({
                        'file': file,
                        'error': result['error']
                    })
                    logger.error(f"❌ Failed to extract student response: {Path(file).name}")
            
            materials['metadata']['student_responses_count'] = len(materials['student_responses'])
            
            # Final check
            if not materials['metadata']['question_loaded']:
                return {
                    'assignment_name': assignment_name,
                    'error': 'Failed to load question text',
                    'metadata': materials['metadata']
                }
            
            if materials['metadata']['solution_keys_count'] == 0:
                return {
                    'assignment_name': assignment_name,
                    'error': 'Failed to load any solution keys',
                    'metadata': materials['metadata']
                }
            
            logger.info(f"✅ Assignment materials loaded successfully: {assignment_name}")
            logger.info(f"   Question: {len(materials['question'])} chars")
            logger.info(f"   Solution keys: {materials['metadata']['solution_keys_count']}")
            logger.info(f"   Student responses: {materials['metadata']['student_responses_count']}")
            
            return materials
            
        except Exception as e:
            logger.error(f"Error loading assignment materials: {e}")
            return {
                'assignment_name': assignment_name,
                'error': str(e)
            }
    
    def upload_file(self, file_path: str, s3_key: str) -> Dict[str, Any]:
        """
        Upload a file to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 object key
            
        Returns:
            Upload result dictionary
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f"File not found: {file_path}"
                }
            
            logger.info(f"Uploading: {file_path} -> s3://{self.bucket_name}/{s3_key}")
            
            self.client.upload_file(file_path, self.bucket_name, s3_key)
            
            logger.info(f"✅ File uploaded successfully: {s3_key}")
            
            return {
                'success': True,
                'bucket': self.bucket_name,
                'key': s3_key,
                'message': 'File uploaded successfully'
            }
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def download_file(self, s3_key: str, local_path: str) -> Dict[str, Any]:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save to
            
        Returns:
            Download result dictionary
        """
        try:
            # Create directories if needed
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            logger.info(f"Downloading: s3://{self.bucket_name}/{s3_key} -> {local_path}")
            
            self.client.download_file(self.bucket_name, s3_key, local_path)
            
            logger.info(f"✅ File downloaded successfully: {s3_key}")
            
            return {
                'success': True,
                'bucket': self.bucket_name,
                'key': s3_key,
                'local_path': local_path,
                'message': 'File downloaded successfully'
            }
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def __str__(self) -> str:
        """String representation of the client"""
        return f"S3Client(bucket={self.bucket_name}, region={self.region_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"S3Client(bucket_name='{self.bucket_name}', region='{self.region_name}')"