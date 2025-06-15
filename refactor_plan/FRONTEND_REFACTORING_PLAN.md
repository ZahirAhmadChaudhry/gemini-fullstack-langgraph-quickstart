# Frontend Refactoring Plan: Research Assistant → Sustainability Analysis Interface

**Date**: 2025-06-15  
**Status**: Implementation Ready  
**Backend**: ✅ Complete (Enhanced prompts + 3-stage analysis pipeline)  
**Frontend**: 🚧 Refactoring in Progress

## Executive Summary

Transform the existing chat-based research assistant frontend into a specialized French sustainability transcript analysis interface. The refactoring maintains the robust LangGraph streaming architecture while completely redesigning the user experience for academic research workflows.

## Current vs Target State

### Current State (Research Assistant)
- **Input**: Single text query via textarea
- **Processing**: Web search with query generation and reflection loops  
- **Output**: Conversational AI responses in chat bubbles
- **UI**: Chat interface with activity timeline showing search steps
- **User Flow**: Question → Search → Conversational Answer

### Target State (Sustainability Analyzer)
- **Input**: French transcript text or JSON file upload
- **Processing**: 3-stage analysis (segmentation → parallel analysis → synthesis)
- **Output**: Structured CSV data with 9 fields per identified tension
- **UI**: Analysis dashboard with results table and export functionality
- **User Flow**: Transcript Upload → Analysis Progress → Structured Results

## Detailed Component Refactoring Plan

### Phase 1: Core Input Interface (Days 1-2)

#### 1.1 InputForm.tsx → TranscriptInputForm.tsx
**Current Issues:**
- Small textarea designed for questions
- Research effort levels (low/medium/high) 
- "Search" button with web search terminology
- No file upload capability

**Refactoring Plan:**
```typescript
// New interface definition
interface TranscriptInputFormProps {
  onSubmit: (transcript: string, options: AnalysisOptions) => void;
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  hasHistory: boolean;
}

interface AnalysisOptions {
  maxSegments: number;        // For free tier management
  analysisModel: string;      // Gemini model selection
  includeMetadata: boolean;   // Use preprocessed data if available
  analysisDepth: 'standard' | 'detailed'; // Analysis complexity
}
```

**Implementation Tasks:**
- [ ] Replace small textarea with large text area (min-height: 200px, max-height: 400px)
- [ ] Add file upload component with drag-and-drop support
- [ ] Support .txt, .json, and .docx file formats
- [ ] Replace effort selector with analysis options:
  - Max segments slider (10-100 segments)
  - Model dropdown (Gemini 2.0/2.5 Flash)
  - Metadata toggle switch
  - Analysis depth selector
- [ ] Update button text: "Search" → "Analyze Transcript"
- [ ] Add French text validation and character count
- [ ] Implement file size limits (max 10MB)

#### 1.2 WelcomeScreen.tsx Updates
**Current Issues:**
- Research assistant branding and examples
- Question-based example prompts
- No guidance for transcript analysis

**Refactoring Plan:**
- [ ] Update title: "Research Assistant" → "French Sustainability Analysis"
- [ ] Replace example questions with transcript examples
- [ ] Add file upload area to welcome screen
- [ ] Include analysis workflow explanation
- [ ] Add supported file format information
- [ ] Update placeholder text and instructions

### Phase 2: Results Display System (Days 3-5)

#### 2.1 Create ResultsTable.tsx (New Component)
**Requirements:**
- Display structured analysis results in tabular format
- Support editing of classification fields
- Export functionality (CSV, JSON)
- Row selection and highlighting
- Sorting and filtering capabilities

**Component Structure:**
```typescript
interface TensionResult {
  id: string;
  conceptsSecondOrder: string;           // Second-order concept
  itemsFirstOrderReformulated: string;  // X vs Y format
  itemsFirstOrderOriginal: string;      // Original text excerpt
  details: string;                      // Full segment text
  synthesis: string;                    // One-line summary
  period: string;                       // 2023/2050
  theme: string;                        // Légitimité/Performance
  codeSpecific: string;                 // Domain code
  imaginaire: string;                   // C/S + IFa/IFr
}

interface ResultsTableProps {
  results: TensionResult[];
  onResultEdit: (id: string, field: string, value: string) => void;
  onExport: (format: 'csv' | 'json') => void;
  isLoading: boolean;
}
```

**Implementation Tasks:**
- [ ] Create responsive table with horizontal scroll
- [ ] Implement editable cells with inline editing
- [ ] Add column sorting and filtering
- [ ] Create export buttons (CSV/JSON download)
- [ ] Add row selection with checkboxes
- [ ] Implement copy functionality for individual cells
- [ ] Add validation indicators for edited fields
- [ ] Create expandable row details view

#### 2.2 Create AnalysisProgress.tsx (New Component)
**Purpose:** Replace ActivityTimeline for sustainability analysis workflow

**Features:**
- [ ] Progress bar showing analysis stages
- [ ] Real-time segment processing updates
- [ ] Token usage and cost monitoring
- [ ] Processing time estimation
- [ ] Error handling and retry options

### Phase 3: State Management Refactoring (Days 6-7)

#### 3.1 App.tsx State Updates
**Current State Schema:**
```typescript
const thread = useStream<{
  messages: Message[];
  initial_search_query_count: number;
  max_research_loops: number;
  reasoning_model: string;
}>
```

**New State Schema:**
```typescript
const thread = useStream<{
  messages: Message[];
  transcript: string;
  preprocessed_data?: any;
  segments?: string[];
  analysis_results?: TensionResult[];
  final_results?: string;
  token_usage?: TokenUsage;
  processing_stats?: ProcessingStats;
}>
```

**Implementation Tasks:**
- [ ] Update useStream configuration for new state fields
- [ ] Modify handleSubmit to send transcript data instead of query
- [ ] Update event processing for analysis workflow
- [ ] Add token usage tracking state
- [ ] Implement results caching and persistence

#### 3.2 Event Processing Updates
**Current Events:**
- `generate_query` → "Generating Search Queries"
- `web_research` → "Web Research" 
- `reflection` → "Reflection"
- `finalize_answer` → "Finalizing Answer"

**New Events:**
- `segment_transcript` → "Segmenting Transcript"
- `analyze_segment` → "Analyzing Segment X/N"
- `finalize_output` → "Synthesizing Results"

### Phase 4: UI/UX Enhancements (Days 8-10)

#### 4.1 ChatMessagesView.tsx → AnalysisView.tsx
**Transformation Plan:**
- [ ] Remove chat bubble interface
- [ ] Replace with analysis dashboard layout
- [ ] Add results table as primary content
- [ ] Keep activity timeline as sidebar component
- [ ] Add export and sharing controls
- [ ] Implement result validation interface

#### 4.2 Navigation and Layout Updates
- [ ] Update page title and meta tags
- [ ] Add breadcrumb navigation
- [ ] Create analysis workflow steps indicator
- [ ] Add help tooltips and documentation links
- [ ] Implement responsive design for mobile viewing

## Technical Implementation Details

### File Upload Implementation
```typescript
// File upload handler
const handleFileUpload = async (file: File) => {
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    throw new Error('File too large');
  }
  
  const text = await file.text();
  setTranscriptText(text);
  
  // Detect if JSON format (preprocessed data)
  try {
    const jsonData = JSON.parse(text);
    setPreprocessedData(jsonData);
  } catch {
    // Plain text transcript
  }
};
```

### Results Table Implementation
```typescript
// Editable cell component
const EditableCell = ({ value, onSave, type }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value);
  
  const handleSave = () => {
    onSave(editValue);
    setIsEditing(false);
  };
  
  return isEditing ? (
    <input 
      value={editValue}
      onChange={(e) => setEditValue(e.target.value)}
      onBlur={handleSave}
      onKeyPress={(e) => e.key === 'Enter' && handleSave()}
    />
  ) : (
    <span onClick={() => setIsEditing(true)}>{value}</span>
  );
};
```

### CSV Export Implementation
```typescript
const exportToCSV = (results: TensionResult[]) => {
  const headers = [
    'Concepts de 2nd ordre',
    'Items de 1er ordre reformulé', 
    'Items de 1er ordre (intitulé d\'origine)',
    'Détails',
    'Synthèse',
    'Période',
    'Thème', 
    'Code spé',
    'Imaginaire'
  ];
  
  const csvContent = [
    headers.join(','),
    ...results.map(row => [
      row.conceptsSecondOrder,
      row.itemsFirstOrderReformulated,
      row.itemsFirstOrderOriginal,
      `"${row.details.replace(/"/g, '""')}"`,
      row.synthesis,
      row.period,
      row.theme,
      row.codeSpecific,
      row.imaginaire
    ].join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `sustainability-analysis-${Date.now()}.csv`;
  a.click();
};
```

## API Integration Testing Plan

### Backend Compatibility Verification
- [ ] Test transcript submission via existing LangGraph API
- [ ] Verify streaming events work with new node names
- [ ] Confirm state field mapping works correctly
- [ ] Test large file handling (300+ pages)
- [ ] Validate structured output parsing

### Integration Test Scenarios
1. **Small Transcript Test** (< 1000 words)
2. **Medium Transcript Test** (5000-10000 words) 
3. **Large Transcript Test** (50000+ words)
4. **JSON Preprocessed Data Test**
5. **Error Handling Test** (invalid files, network issues)

## Success Metrics

### Functional Requirements
- [ ] Users can upload French transcripts via file or text input
- [ ] Analysis results display in structured table format
- [ ] Results are editable and exportable to CSV
- [ ] Real-time progress updates during analysis
- [ ] Token usage and cost monitoring works
- [ ] Error handling provides clear user feedback

### Performance Requirements  
- [ ] File upload handles up to 10MB files
- [ ] Table renders smoothly with 100+ results
- [ ] Analysis progress updates in real-time
- [ ] Export functionality works for large datasets
- [ ] Mobile responsive design functions properly

### User Experience Requirements
- [ ] Intuitive workflow from upload to results
- [ ] Clear visual feedback during processing
- [ ] Professional academic research interface
- [ ] Comprehensive help and documentation
- [ ] Accessibility compliance (WCAG 2.1)

## Risk Mitigation Strategies

### Technical Risks
- **Large File Performance**: Implement chunking and lazy loading
- **Browser Memory**: Use virtual scrolling for large result sets
- **API Compatibility**: Thorough testing with backend state changes
- **Real-time Updates**: Fallback polling if WebSocket fails

### User Experience Risks  
- **Learning Curve**: Comprehensive onboarding and help system
- **Data Loss**: Auto-save functionality and session persistence
- **Export Issues**: Multiple export formats and error recovery
- **Mobile Usability**: Responsive design with touch-friendly controls

## Implementation Timeline

**Week 1: Core Functionality**
- Days 1-2: Input interface refactoring
- Days 3-5: Results display system
- Days 6-7: State management updates

**Week 2: Enhancement & Testing**
- Days 8-10: UI/UX improvements
- Days 11-12: Integration testing
- Days 13-14: Performance optimization and bug fixes

**Total Estimated Effort**: 2 weeks full-time development

## Next Steps

1. **Begin Phase 1**: Start with InputForm.tsx refactoring
2. **Create Test Data**: Prepare sample transcripts for testing
3. **Backend Coordination**: Ensure streaming events align with frontend expectations
4. **Documentation**: Update user guides and technical documentation
5. **Deployment Planning**: Prepare staging environment for testing

This plan provides a comprehensive roadmap for transforming the research assistant into a specialized sustainability analysis tool while maintaining the robust technical foundation and user experience quality.
