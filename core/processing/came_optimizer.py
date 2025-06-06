from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
import camelot


class CameOptimizer:
    def __init__(self, 
                 target_accuracy: float = 80.0,
                 target_whitespace: float = 30.0,
                 max_iterations: int = 3):
        self.target_accuracy = target_accuracy
        self.target_whitespace = target_whitespace
        self.max_iterations = max_iterations
        
    def optimize_table(self, pdf_path: str, initial_params: Dict):
        """
        Optimize table extraction parameters
        
        Args:
            pdf_path: Path to PDF file
            initial_params: Initial parameters
            
        Returns:
            List[camelot.Table]: List of optimized tables
        """
        # First extraction
        initial_tables = self._extract_tables(pdf_path, initial_params)
        if not initial_tables:
            return initial_tables
        print(f"Initial number of tables: {len(initial_tables)}")
        print(f"Target accuracy: {self.target_accuracy}, Target whitespace: {self.target_whitespace}")
        
        # Check each table's accuracy and whitespace
        optimized_tables = []
        for table in initial_tables:
            # Check if current table meets requirements
            if self._is_acceptable(table.accuracy, table.whitespace):
                print("Table meets requirements, skipping optimization")
                optimized_tables.append(table)
                continue
            
            # Adjust parameters and re-extract table
            refined_tables = self._refine_table(pdf_path, table, initial_params)
            # Handle return value which could be single table or list of tables
            if isinstance(refined_tables, list):
                optimized_tables.extend(refined_tables)
            else:
                optimized_tables.append(refined_tables)
        
        return optimized_tables
            
    def _extract_tables(self, pdf_path: str, params: Dict):
        """Extract tables with given parameters"""
        try:
            return camelot.read_pdf(pdf_path, **params)
        except Exception as e:
            logging.error(f"Table extraction failed: {str(e)}")
            return None      
    
    
    def _refine_table(self, pdf_path: str, table, params: Dict):
        """Refine individual table parameters"""              
        # Initialize adjustment counter
        adjustments = 0
        best_table = table
        best_params = params.copy()
        
        # Subsequent parameter adjustments only within table's page and region
        print(f"Table region: {table._bbox}")
        expanded_table = self.convert_and_expand_bbox(table, 10)
        best_params['table_regions'] = [f"{expanded_table[0]},{expanded_table[1]},{expanded_table[2]},{expanded_table[3]}"]
        best_params['pages'] = str(table.page)

        b_process_background = False
        
        while adjustments < 3:          
            print(f"Parameters before optimization: {best_params}")
            print(f"Adjustment {adjustments + 1}")
            
            try:
                if params['flavor'] == 'lattice':
                    if not b_process_background:
                        # Save original parameters
                        original_params = best_params.copy()
                        best_params['process_background'] = True
                        b_process_background = True
                    else:
                        # Adjust line_scale based on table structure
                        best_params.update(self.optimize_lattice_params(best_table))
                    
                elif params['flavor'] == 'stream':
                    # Save current table_regions and pages parameters
                    current_regions = best_params.get('table_regions')
                    current_pages = best_params.get('pages')
                    # Update other parameters
                    best_params.update(self.optimize_stream_params(best_table))
                    # Restore table_regions and pages parameters
                    best_params['table_regions'] = current_regions
                    best_params['pages'] = current_pages

                elif params['flavor'] == 'network':
                    best_params.update(self.optimize_network_params(best_table))

                elif params['flavor'] == 'hybrid':
                    best_params.update(self.optimize_hybrid_params(best_table))
                
                print(f"Parameters after optimization: {best_params}")
                new_table = self._extract_tables(pdf_path, best_params)
                print(f"Number of newly extracted tables: {len(new_table)}")
                
                if not new_table:
                    if params['flavor'] == 'lattice' and best_params.get("process_background"):
                        # If process_background fails, restore original parameters
                        best_params = original_params.copy()
                        print("process_background failed, restoring original parameters")
                        continue
                    else:
                        # If no tables extracted, return original table
                        print("No tables extracted after parameter adjustment, returning original table")
                        return best_table
                
                # Handle multiple tables case
                if len(new_table) > 1:
                    print(f"Detected {len(new_table)} tables in specified region")
                    # 1. First check if any tables meet requirements
                    acceptable_tables = [t for t in new_table if self._is_acceptable(t.accuracy, t.whitespace)]
                    if acceptable_tables:
                        # If there are acceptable tables, choose the one with highest accuracy
                        best_new_table = max(acceptable_tables, key=lambda t: t.accuracy)
                        print(f"Selected table with highest accuracy: accuracy {best_new_table.accuracy}, whitespace {best_new_table.whitespace}")
                        return best_new_table
                    
                    # 2. If no acceptable tables, choose the one most similar to original
                    # Calculate overlap with original table
                    def calculate_overlap(t):
                        # Get table bounding box
                        t_bbox = t._bbox
                        orig_bbox = table._bbox
                        # Calculate overlap area
                        x_overlap = max(0, min(t_bbox[2], orig_bbox[2]) - max(t_bbox[0], orig_bbox[0]))
                        y_overlap = max(0, min(t_bbox[3], orig_bbox[3]) - max(t_bbox[1], orig_bbox[1]))
                        overlap_area = x_overlap * y_overlap
                        # Calculate original table area
                        orig_area = (orig_bbox[2] - orig_bbox[0]) * (orig_bbox[3] - orig_bbox[1])
                        return overlap_area / orig_area if orig_area > 0 else 0
                    
                    # Select table with highest overlap
                    best_new_table = max(new_table, key=calculate_overlap)
                    print(f"Selected table with highest overlap: overlap {calculate_overlap(best_new_table):.2f}")
                    return best_new_table
                
                # If only one table, use it directly
                new_table = new_table[0]
                
                # Check metrics after process_background
                print(f"New table accuracy: {new_table.accuracy}, whitespace: {new_table.whitespace}")
                if self._is_acceptable(new_table.accuracy, new_table.whitespace):
                    return new_table
                
                # Only lattice mode can use process_background
                # If process_background gives better results, enable it and adjust parameters
                if params['flavor'] == 'lattice' and best_params["process_background"]:
                    if self._is_better(new_table.accuracy, new_table.whitespace, table.accuracy, table.whitespace):
                        best_table = new_table
                    else:
                        # If not better, restore original parameters
                        best_table = table
                        best_params["process_background"] = False
                print(f"Current parameters: {best_params}")
                adjustments += 1

            except Exception as e:
                print(f"Parameter adjustment failed: {str(e)}")
                return best_table               
        return best_table
    
    def _is_acceptable(self, accuracy, whitespace) -> bool:
        """Check if metrics meet requirements"""
        print(f"Current accuracy: {accuracy}, whitespace: {whitespace}")
        
        return (accuracy >= self.target_accuracy)# and 
                #whitespace <= self.target_whitespace)
                
    def _is_better(self, accuracy, whitespace, old_accuracy, old_whitespace) -> bool:
        """Compare two sets of metrics"""
        if accuracy > old_accuracy:
            return True
        #elif accuracy == old_accuracy:
         #   return whitespace < old_whitespace
        return False       
        
    def analyze_table_structure(self, table) -> Dict:
        """Analyze table structure features
        
        Args:
            table: Camelot table object
            
        Returns:
            Dict: Table structure features
        """
        features = {
            'cell_heights': [],  # Cell heights
            'cell_widths': [],   # Cell widths
            'text_heights': [],  # Text heights
        }
        
        # 1. Analyze cell structure
        if hasattr(table, 'cells') and table.cells:
            for i, row in enumerate(table.cells):
                for j, cell in enumerate(row):
                    # Record cell dimensions
                    features['cell_heights'].append(abs(cell.lt[1] - cell.rb[1]))
                    features['cell_widths'].append(abs(cell.rb[0] - cell.lt[0]))
                    
        # Analyze text box structure in cells
        if hasattr(table, '_text'):
            for text in table._text:
                # Record text height
                x1, y1, x2, y2 = text
                features['text_heights'].append(int(y2 - y1))

        return features
    
    def optimize_lattice_params(self, table) -> Dict:
        """Optimize Lattice mode parameters"""
        features = self.analyze_table_structure(table)
        params = {}
        
        # 1. Adjust line_scale
        if features['cell_heights'] and features['text_heights']:
            min_cell_h = np.min(features['cell_heights']) if features['cell_heights'] else 0
            min_text_h = np.min(features['text_heights']) if features['text_heights'] else 0
            params['line_scale'] = min(int(min_cell_h), int(min_text_h))
        
        return params
    
    def optimize_stream_params(self, table) -> Dict:
        """Optimize Stream mode parameters"""
        features = self.analyze_table_structure(table)
        params = {}
        
        # 1. Adjust edge_tol
        if features['cell_heights']:
            params['edge_tol'] = max(30, min(features['cell_heights']))
        
        # 2. Adjust row_tol
        if features['text_heights']:
            params['row_tol'] = np.min(features['text_heights'])
        
        return params

    def optimize_network_params(self, table):
        """Automatically adjust Network mode parameters based on text network analysis"""
        params = {}
        if not table.parse_details:
            return None
            
        # 1. Analyze text alignment network
        network_searches = table.parse_details["network_searches"]
        all_gaps = []
        
        for network in network_searches:
            gaps_hv = network.compute_plausible_gaps()
            if gaps_hv:
                h_gap, v_gap = gaps_hv
                all_gaps.append((h_gap, v_gap))
                
        if not all_gaps:
            return None
            
        # 2. Adjust edge_tol
        # Theory: edge_tol based on 75th percentile of vertical gaps
        v_gaps = [g[1] for g in all_gaps]
        params['edge_tol'] = int(np.percentile(v_gaps, 75))
        
        # 3. Adjust row_tol and column_tol
        # Theory: based on minimum gaps in horizontal and vertical directions
        h_gaps = [g[0] for g in all_gaps]
        params['row_tol'] = max(2, int(min(v_gaps)/4))
        params['column_tol'] = max(0, int(min(h_gaps)/4))
        
        return params
    
    def optimize_hybrid_params(self, table):
        """Combine Lattice and Network analysis results to adjust Hybrid mode parameters"""
        params = {}
        if not table.parse_details:
            return None
            
        # 1. Get parameter suggestions from both methods
        network_params = self.optimize_network_params(table)
        lattice_params = self.optimize_lattice_params(table)        
        
        if not (lattice_params and network_params):
            return None
            
        params['line_scale'] = lattice_params['line_scale']
        params['line_tol'] = lattice_params['line_tol']
        params['joint_tol'] = lattice_params['joint_tol']
        params['row_tol'] = network_params['row_tol']
        params['column_tol'] = network_params['column_tol']
        params['edge_tol'] = network_params['edge_tol']        
        
        return params
    
    def convert_and_expand_bbox(self, table, percentage: int) -> List[float]:
        """Convert and expand table bounding box
        
        Args:
            table: Initial extracted table object
            percentage: Expansion percentage
            
        Returns:
            List[float]: Converted and expanded bounding box [x1, y1, x2, y2]
        """
        if not hasattr(table, '_bbox') or not table._bbox:
            return None
            
        # 1. Get PDF page height (key step)
        pdf_height = table.pdf_size[1]
        if not pdf_height:
            return None
            
        print(f"Original bounding box: {table._bbox}")
        # 2. Convert coordinate system
        bbox = self._convert_bbox_coordinates(table._bbox, pdf_height)
        print(f"Converted bounding box: {bbox}")
        # 3. Expand bounding box
        expanded_bbox = self._expand_bbox(bbox, table.pdf_size[0], pdf_height, percentage)
        
        return expanded_bbox
    
    def _convert_bbox_coordinates(self, bbox: List[float], pdf_height: float) -> List[float]:
        """Convert bounding box coordinates
        
        Args:
            bbox: Original bounding box coordinates [x1, y1, x2, y2]
            pdf_height: PDF page height
            
        Returns:
            List[float]: Converted coordinates [x1, y1, x2, y2]
        """
        # Camelot uses PDF coordinate system (origin at bottom-left)
        # table._bbox uses image coordinate system (origin at top-left)
        # Need to convert y coordinates
        x1, y1, x2, y2 = bbox
        return [
            x1,                    # x1 unchanged
            pdf_height - y2,       # convert y1
            x2,                    # x2 unchanged
            pdf_height - y1        # convert y2
        ]
    
    def _expand_bbox(self, bbox, pdf_width, pdf_height, percentage: int) -> Tuple[float, float, float, float]:
        """Expand table region"""
        # Expand table boundaries by percentage
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        new_x0 = max(0, x0 - width * percentage / 100)
        new_y0 = max(0, y0 - height * percentage / 100)
        new_x1 = min(x1 + width * percentage / 100, pdf_width)
        new_y1 = min(y1 + height * percentage / 100, pdf_height)
        new_bbox = (new_x0, new_y0, new_x1, new_y1)
        return new_bbox
    
    
  
    

    
    
