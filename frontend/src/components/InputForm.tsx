import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, Send, StopCircle, Zap, FileText, X } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

// New interface definitions for sustainability analysis
export interface AnalysisOptions {
  maxSegments: number;
  analysisModel: string;
  includeMetadata: boolean;
  analysisDepth: 'standard' | 'detailed';
}

interface TranscriptInputFormProps {
  onSubmit: (transcript: string, options: AnalysisOptions) => void; // Kept options here as InputForm collects them
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  onCancel?: () => void;
}

export const TranscriptInputForm: React.FC<TranscriptInputFormProps> = ({
  onSubmit,
  onFileUpload,
  isLoading,
  onCancel,
}) => {
  const [transcriptText, setTranscriptText] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [analysisOptions, setAnalysisOptions] = useState<AnalysisOptions>({
    maxSegments: 50,
    analysisModel: "gemini-2.0-flash",
    includeMetadata: false,
    analysisDepth: 'standard'
  });
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!transcriptText.trim()) return;
    onSubmit(transcriptText, analysisOptions); // InputForm still submits with options
    setTranscriptText("");
    setUploadedFile(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleFileUpload = async (file: File) => {
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      alert('File too large. Maximum size is 10MB.');
      return;
    }

    const allowedTypes = ['text/plain', 'application/json', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.txt') && !file.name.endsWith('.json') && !file.name.endsWith('.docx')) {
      alert('Unsupported file type. Please upload .txt, .json, or .docx files.');
      return;
    }

    try {
      const text = await file.text();
      setTranscriptText(text);
      setUploadedFile(file);

      // Detect if JSON format (preprocessed data)
      try {
        JSON.parse(text); // Check if it's valid JSON
        setAnalysisOptions(prev => ({ ...prev, includeMetadata: true }));
      } catch {
        // Plain text transcript
        setAnalysisOptions(prev => ({ ...prev, includeMetadata: false }));
      }

      onFileUpload(file);
    } catch (error) {
      alert('Error reading file. Please try again.');
      console.error('File upload error:', error);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const removeFile = () => {
    setUploadedFile(null);
    setTranscriptText("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isSubmitDisabled = !transcriptText.trim() || isLoading;
  const characterCount = transcriptText.length;
  const isValidFrenchText = transcriptText.length > 100; // Basic validation

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4 p-4">
      {/* File Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
          isDragOver
            ? "border-blue-400 bg-blue-500/10"
            : "border-neutral-600 hover:border-neutral-500"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".txt,.json,.docx"
          onChange={handleFileInputChange}
          className="hidden"
        />

        {uploadedFile ? (
          <div className="flex items-center justify-center gap-2">
            <FileText className="h-5 w-5 text-green-400" />
            <span className="text-neutral-300">{uploadedFile.name}</span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={removeFile}
              className="text-red-400 hover:text-red-300"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        ) : (
          <div>
            <Upload className="h-8 w-8 text-neutral-400 mx-auto mb-2" />
            <p className="text-neutral-300 mb-1">
              Drop your French transcript here or{" "}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="text-blue-400 hover:text-blue-300 underline"
              >
                browse files
              </button>
            </p>
            <p className="text-xs text-neutral-500">
              Supports .txt, .json, .docx files (max 10MB)
            </p>
          </div>
        )}
      </div>

      {/* Text Input Area */}
      <div className="bg-neutral-700 rounded-lg p-4">
        <Textarea
          value={transcriptText}
          onChange={(e) => setTranscriptText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Paste your French sustainability transcript here..."
          className="w-full text-neutral-100 placeholder-neutral-500 resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none bg-transparent min-h-[200px] max-h-[400px]"
          rows={8}
        />
        <div className="flex justify-between items-center mt-2 text-xs">
          <span className={`${isValidFrenchText ? 'text-green-400' : 'text-neutral-500'}`}>
            {characterCount} characters {isValidFrenchText ? '✓' : '(minimum 100)'}
          </span>
        </div>
      </div>
      {/* Analysis Options */}
      <div className="bg-neutral-800 rounded-lg p-4 space-y-4">
        <h3 className="text-sm font-medium text-neutral-300 mb-3">Analysis Options</h3>

        {/* Max Segments Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm text-neutral-400">Max Segments</Label>
            <span className="text-sm text-neutral-300">{analysisOptions.maxSegments}</span>
          </div>
          <Slider
            value={[analysisOptions.maxSegments]}
            onValueChange={(value) =>
              setAnalysisOptions(prev => ({ ...prev, maxSegments: value[0] }))
            }
            max={100}
            min={10}
            step={10}
            className="w-full"
          />
        </div>

        {/* Model Selection */}
        <div className="space-y-2">
          <Label className="text-sm text-neutral-400">Analysis Model</Label>
          <Select
            value={analysisOptions.analysisModel}
            onValueChange={(value) =>
              setAnalysisOptions(prev => ({ ...prev, analysisModel: value }))
            }
          >
            <SelectTrigger className="bg-neutral-700 border-neutral-600 text-neutral-300">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300">
              <SelectItem value="gemini-2.0-flash">
                <div className="flex items-center">
                  <Zap className="h-4 w-4 mr-2 text-yellow-400" /> Gemini 2.0 Flash
                </div>
              </SelectItem>
              <SelectItem value="gemini-2.5-flash">
                <div className="flex items-center">
                  <Zap className="h-4 w-4 mr-2 text-orange-400" /> Gemini 2.5 Flash
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Analysis Depth */}
        <div className="space-y-2">
          <Label className="text-sm text-neutral-400">Analysis Depth</Label>
          <Select
            value={analysisOptions.analysisDepth}
            onValueChange={(value: 'standard' | 'detailed') =>
              setAnalysisOptions(prev => ({ ...prev, analysisDepth: value }))
            }
          >
            <SelectTrigger className="bg-neutral-700 border-neutral-600 text-neutral-300">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-neutral-700 border-neutral-600 text-neutral-300">
              <SelectItem value="standard">Standard Analysis</SelectItem>
              <SelectItem value="detailed">Detailed Analysis</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Metadata Toggle */}
        <div className="flex items-center justify-between">
          <Label className="text-sm text-neutral-400">Include Metadata</Label>
          <Switch
            checked={analysisOptions.includeMetadata}
            onCheckedChange={(checked) =>
              setAnalysisOptions(prev => ({ ...prev, includeMetadata: checked }))
            }
          />
        </div>
      </div>

      {/* Buttons */}
      <div className="flex flex-col sm:flex-row gap-2 mt-4">
        {isLoading ? (
          <Button
            type="button"
            onClick={onCancel} // This uses the onCancel prop
            variant="destructive"
            className="w-full"
            disabled={!onCancel} // Disable if onCancel is not provided
          >
            <StopCircle className="mr-2 h-4 w-4" /> Cancel
          </Button>
        ) : (
          <Button type="submit" disabled={isSubmitDisabled} className="w-full">
            <Send className="mr-2 h-4 w-4" /> Submit Analysis
          </Button>
        )}
      </div>
    </form>
  );
};

// Export both for backward compatibility during transition
export const InputForm = TranscriptInputForm;
