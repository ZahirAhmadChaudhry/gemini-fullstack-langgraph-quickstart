import React from "react";
import { TranscriptInputForm, AnalysisOptions } from "./InputForm";
import { FileText, Zap, Target } from "lucide-react";

interface WelcomeScreenProps {
  handleSubmit: (transcript: string, options: AnalysisOptions) => void; // onSubmit here still expects options from InputForm
  onFileUpload: (file: File) => void;
  isLoading: boolean;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onFileUpload,
  isLoading,
}) => (
  <div className="flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-4xl mx-auto gap-6">
    <div>
      <h1 className="text-4xl md:text-5xl font-semibold text-neutral-100 mb-3">
        French Sustainability Analysis
      </h1>
      <p className="text-lg md:text-xl text-neutral-400 mb-4">
        Analyze French transcripts for sustainability tensions and paradoxes
      </p>

      {/* Feature highlights */}
      <div className="flex flex-wrap justify-center gap-4 mb-6">
        <div className="flex items-center gap-2 text-sm text-neutral-300 bg-neutral-800 px-3 py-2 rounded-lg">
          <FileText className="h-4 w-4 text-blue-400" />
          <span>Upload transcripts or paste text</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-neutral-300 bg-neutral-800 px-3 py-2 rounded-lg">
          <Zap className="h-4 w-4 text-yellow-400" />
          <span>AI-powered segmentation</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-neutral-300 bg-neutral-800 px-3 py-2 rounded-lg">
          <Target className="h-4 w-4 text-green-400" />
          <span>Structured CSV results</span>
        </div>
      </div>
    </div>

    <div className="w-full mt-4">
      <TranscriptInputForm
        onSubmit={handleSubmit}
        onFileUpload={onFileUpload}
        isLoading={isLoading}
      />
    </div>

    {/* Example section */}
    <div className="text-left w-full max-w-2xl">
      <h3 className="text-sm font-medium text-neutral-300 mb-2">Example Analysis Input:</h3>
      <div className="bg-neutral-800 rounded-lg p-3 text-xs text-neutral-400">
        <p className="italic">
          "D'un côté, nous devons réduire notre empreinte carbone pour préserver l'environnement,
          mais d'autre part, nous devons maintenir notre croissance économique pour assurer
          l'emploi et la prospérité..."
        </p>
      </div>
    </div>

    <p className="text-xs text-neutral-500">
      Powered by Google Gemini 2.0 Flash and LangChain LangGraph
    </p>
  </div>
);
