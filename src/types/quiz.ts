export interface Question {
  id: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  reference: string;
}

export interface QuizState {
  currentQuestionIndex: number;
  score: number;
  answers: number[];
  showResults: boolean;
}
