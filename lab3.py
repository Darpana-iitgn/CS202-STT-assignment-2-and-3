import pandas as pd
import numpy as np
import tempfile
import os
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze
import torch
from transformers import AutoTokenizer, AutoModel
try:
    from sacrebleu import sentence_bleu
except ImportError:
    from sacrebleu.metrics import BLEU
    sentence_bleu = BLEU()
import warnings
warnings.filterwarnings('ignore')

class BugContextAnalyzer:
    def __init__(self, csv_path):
        """
        Initialize the analyzer with the CSV file path
        
        Args:
            csv_path: Path to the CSV file from Lab 2
        """
        self.csv_path = csv_path
        self.df = None
        
        # Initialize CodeBERT for semantic similarity
        print("Loading CodeBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = AutoModel.from_pretrained('microsoft/codebert-base')
        self.model.eval()
        print("CodeBERT model loaded successfully!")
        
    def load_data(self):
        """Load and validate the CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Check required columns
            required_cols = ['Hash', 'Message', 'Filename', 'Source Code (before)', 
                           'Source Code (current)', 'Diff', 'LLM Inference (fix type)', 
                           'Rectified Message']
            
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                print(f"Available columns: {list(self.df.columns)}")
                
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def compute_baseline_statistics(self):
        """Compute baseline descriptive statistics"""
        print("\n=== BASELINE STATISTICS ===")
        
        # Total commits and files
        total_commits = self.df['Hash'].nunique()
        total_files = len(self.df)
        print(f"Total unique commits: {total_commits}")
        print(f"Total files: {total_files}")
        
        # Average files per commit
        files_per_commit = self.df.groupby('Hash').size()
        avg_files_per_commit = files_per_commit.mean()
        print(f"Average files per commit: {avg_files_per_commit:.2f}")
        
        # Distribution of fix types
        if 'LLM Inference (fix type)' in self.df.columns:
            fix_types = self.df['LLM Inference (fix type)'].value_counts()
            print(f"\nFix type distribution:")
            for fix_type, count in fix_types.items():
                print(f"  {fix_type}: {count}")
        
        # Most frequent file extensions
        if 'Filename' in self.df.columns:
            extensions = self.df['Filename'].str.extract(r'\.([^.]+)$')[0].value_counts()
            print(f"\nMost frequent file extensions:")
            for ext, count in extensions.head().items():
                print(f"  .{ext}: {count}")
    
    def safe_write_temp_file(self, code, suffix='.py'):
        """Safely write code to a temporary file"""
        if pd.isna(code) or not isinstance(code, str) or not code.strip():
            return None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
                f.write(code)
                return f.name
        except Exception as e:
            print(f"Error writing temp file: {e}")
            return None
    
    def compute_radon_metrics(self, code, filename=""):
        """Compute radon metrics for given code"""
        if pd.isna(code) or not isinstance(code, str) or not code.strip():
            return {'MI': 0, 'CC': 0, 'LOC': 0}
        
        # Check if this is Python code based on file extension
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        # Only compute radon metrics for Python files
        if file_ext not in ['py', 'pyx', 'pyw']:
            # For non-Python files, compute basic metrics
            lines = code.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            return {
                'MI': 100,  # Default good maintainability for non-code files
                'CC': 1,    # Minimal complexity
                'LOC': len(non_empty_lines)
            }
        
        temp_file = None
        try:
            # Write code to temporary file with .py extension
            temp_file = self.safe_write_temp_file(code, '.py')
            if temp_file is None:
                return {'MI': 0, 'CC': 0, 'LOC': 0}
            
            # Read the file content for analysis
            with open(temp_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Try to parse as Python - if it fails, treat as non-code
            try:
                compile(file_content, temp_file, 'exec')
            except SyntaxError:
                # If it's not valid Python, return basic metrics
                lines = file_content.strip().split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                return {
                    'MI': 100,  # Default good maintainability
                    'CC': 1,    # Minimal complexity
                    'LOC': len(non_empty_lines)
                }
            
            # Compute radon metrics for valid Python code
            mi_score = mi_visit(file_content, multi=True)
            cc_results = cc_visit(file_content)
            raw_metrics = analyze(file_content)
            
            # Extract values
            mi_value = mi_score if isinstance(mi_score, (int, float)) else 0
            cc_value = sum(item.complexity for item in cc_results) if cc_results else 0
            loc_value = raw_metrics.loc if raw_metrics else 0
            
            return {'MI': mi_value, 'CC': cc_value, 'LOC': loc_value}
            
        except Exception as e:
            # Fallback to basic line counting
            lines = code.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            return {
                'MI': 100,  # Default value
                'CC': 1,    # Default value
                'LOC': len(non_empty_lines)
            }
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def compute_semantic_similarity(self, code1, code2):
        """Compute semantic similarity using CodeBERT"""
        if pd.isna(code1) or pd.isna(code2) or not code1.strip() or not code2.strip():
            return 0.0
        
        try:
            # Tokenize both code snippets
            inputs1 = self.tokenizer(code1, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            inputs2 = self.tokenizer(code2, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)
                
                # Use [CLS] token embeddings
                emb1 = outputs1.last_hidden_state[:, 0, :].squeeze()
                emb2 = outputs2.last_hidden_state[:, 0, :].squeeze()
                
                # Compute cosine similarity
                similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                return max(0.0, similarity)  # Ensure non-negative
                
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def compute_token_similarity(self, code1, code2):
        """Compute token similarity using BLEU"""
        if pd.isna(code1) or pd.isna(code2) or not code1.strip() or not code2.strip():
            return 0.0
        
        try:
            # Simple tokenization by splitting on whitespace and special characters
            import re
            
            def tokenize_code(code):
                # Split on whitespace and common programming delimiters
                tokens = re.findall(r'\w+|[^\w\s]', code)
                return [token.strip() for token in tokens if token.strip()]
            
            tokens1 = tokenize_code(code1)
            tokens2 = tokenize_code(code2)
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Convert to strings for BLEU
            reference = ' '.join(tokens1)
            hypothesis = ' '.join(tokens2)
            
            # Compute BLEU score - handle different sacrebleu versions
            try:
                # Try newer sacrebleu API
                if hasattr(sentence_bleu, 'sentence_score'):
                    bleu_score = sentence_bleu.sentence_score(hypothesis, [reference]).score / 100.0
                else:
                    # Try older API
                    bleu_score = sentence_bleu(reference, [hypothesis]).score / 100.0
            except:
                # Fallback: simple token overlap similarity
                set1, set2 = set(tokens1), set(tokens2)
                if len(set1) == 0 and len(set2) == 0:
                    return 1.0
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                bleu_score = intersection / union if union > 0 else 0.0
            
            return min(1.0, max(0.0, bleu_score))  # Ensure between 0 and 1
            
        except Exception as e:
            print(f"Error computing token similarity: {e}")
            return 0.0
    
    def classify_fixes(self, semantic_sim, token_sim, 
                      semantic_threshold=0.80, token_threshold=0.75):
        """Classify fixes as Major or Minor based on similarity thresholds"""
        semantic_class = "Minor" if semantic_sim >= semantic_threshold else "Major"
        token_class = "Minor" if token_sim >= token_threshold else "Major"
        classes_agree = "YES" if semantic_class == token_class else "NO"
        
        return semantic_class, token_class, classes_agree
    
    def process_all_metrics(self):
        """Process all metrics for the dataset"""
        print("\n=== PROCESSING METRICS ===")
        
        # Initialize new columns
        metrics_cols = ['MI_Before', 'MI_After', 'CC_Before', 'CC_After', 
                       'LOC_Before', 'LOC_After', 'MI_Change', 'CC_Change', 
                       'LOC_Change', 'Semantic_Similarity', 'Token_Similarity',
                       'Semantic_Class', 'Token_Class', 'Classes_Agree']
        
        for col in metrics_cols:
            self.df[col] = 0
        
        # Process each row
        for idx, row in self.df.iterrows():
            print(f"Processing row {idx + 1}/{len(self.df)}...")
            
            code_before = row.get('Source Code (before)', '')
            code_after = row.get('Source Code (current)', '')
            
            # Get filename for context
            filename = row.get('Filename', '')
            
            # Compute radon metrics
            metrics_before = self.compute_radon_metrics(code_before, filename)
            metrics_after = self.compute_radon_metrics(code_after, filename)
            
            # Update dataframe with radon metrics (ensure proper data types)
            self.df.loc[idx, 'MI_Before'] = round(float(metrics_before['MI']), 2)
            self.df.loc[idx, 'MI_After'] = round(float(metrics_after['MI']), 2)
            self.df.loc[idx, 'CC_Before'] = int(metrics_before['CC'])
            self.df.loc[idx, 'CC_After'] = int(metrics_after['CC'])
            self.df.loc[idx, 'LOC_Before'] = int(metrics_before['LOC'])
            self.df.loc[idx, 'LOC_After'] = int(metrics_after['LOC'])
            
            # Compute changes
            mi_change = round(float(metrics_after['MI'] - metrics_before['MI']), 2)
            cc_change = int(metrics_after['CC'] - metrics_before['CC'])
            loc_change = int(metrics_after['LOC'] - metrics_before['LOC'])
            
            self.df.loc[idx, 'MI_Change'] = mi_change
            self.df.loc[idx, 'CC_Change'] = cc_change
            self.df.loc[idx, 'LOC_Change'] = loc_change
            
            # Compute similarity metrics
            semantic_sim = self.compute_semantic_similarity(code_before, code_after)
            token_sim = self.compute_token_similarity(code_before, code_after)
            
            self.df.loc[idx, 'Semantic_Similarity'] = semantic_sim
            self.df.loc[idx, 'Token_Similarity'] = token_sim
            
            # Classify fixes
            sem_class, tok_class, agree = self.classify_fixes(semantic_sim, token_sim)
            self.df.loc[idx, 'Semantic_Class'] = sem_class
            self.df.loc[idx, 'Token_Class'] = tok_class
            self.df.loc[idx, 'Classes_Agree'] = agree
    
    def save_results(self, output_path=None):
        """Save the processed results"""
        if output_path is None:
            output_path = self.csv_path.replace('.csv', '_lab3_results.csv')
        
        self.df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Display summary
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Total processed files: {len(self.df)}")
        print(f"Semantic-Token agreement: {(self.df['Classes_Agree'] == 'YES').sum()}/{len(self.df)}")
        print(f"Major fixes (Semantic): {(self.df['Semantic_Class'] == 'Major').sum()}")
        print(f"Major fixes (Token): {(self.df['Token_Class'] == 'Major').sum()}")
    
    def display_sample_results(self, n=3):
        """Display sample results"""
        print(f"\n=== SAMPLE RESULTS (First {n} rows) ===")
        display_cols = ['Hash', 'Filename', 'MI_Before', 'MI_After', 'MI_Change', 
                       'CC_Before', 'CC_After', 'CC_Change', 
                       'LOC_Before', 'LOC_After', 'LOC_Change',
                       'Semantic_Similarity', 'Token_Similarity', 'Semantic_Class', 
                       'Token_Class', 'Classes_Agree']
        
        available_cols = [col for col in display_cols if col in self.df.columns]
        sample_df = self.df[available_cols].head(n)


def main():
    # Configuration
    CSV_PATH = "output_files/llm_rectified_message.csv"
    
    print("=== Bug Context Analysis Lab 3 ===")
    print(f"Processing file: {CSV_PATH}")
    
    # Initialize analyzer
    analyzer = BugContextAnalyzer(CSV_PATH)
    
    # Load and process data
    if not analyzer.load_data():
        print("Failed to load data. Exiting...")
        return
    
    # Compute baseline statistics
    analyzer.compute_baseline_statistics()
    
    # Process all metrics
    analyzer.process_all_metrics()
    
    # Display sample results
    analyzer.display_sample_results()
    
    # Save results
    analyzer.save_results()
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
    

