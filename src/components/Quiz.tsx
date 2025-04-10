'use client';

import { useState, useRef } from 'react';
import { QuizState } from '../types/quiz';
import { questions } from '../data/quizData';
import * as htmlToImage from 'html-to-image';

interface ScoreHistory {
  date: string;
  score: number;
  totalQuestions: number;
  percentage: number;
}

export default function Quiz() {
  const [state, setState] = useState<QuizState>({
    currentQuestionIndex: 0,
    score: 0,
    answers: [],
    showResults: false,
  });
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showFeedback, setShowFeedback] = useState(false);
  const [scoreHistory, setScoreHistory] = useState<ScoreHistory[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('quizScoreHistory');
      return saved ? JSON.parse(saved) : [];
    }
    return [];
  });
  
  const resultsRef = useRef<HTMLDivElement>(null);

  const currentQuestion = questions[state.currentQuestionIndex];

  const handleAnswer = (selectedOption: number) => {
    setSelectedAnswer(selectedOption);
    setShowFeedback(true);
    const isCorrect = selectedOption === currentQuestion.correctAnswer;
    if (isCorrect) {
      setState(prev => ({
        ...prev,
        score: prev.score + 1,
      }));
    }
  };

  const handleNextQuestion = () => {
    const newAnswers = [...state.answers, selectedAnswer!];
    
    if (state.currentQuestionIndex === questions.length - 1) {
      const newScore: ScoreHistory = {
        date: new Date().toLocaleString(),
        score: state.score,
        totalQuestions: questions.length,
        percentage: (state.score / questions.length) * 100
      };
      
      const newHistory = [...scoreHistory, newScore];
      setScoreHistory(newHistory);
      if (typeof window !== 'undefined') {
        localStorage.setItem('quizScoreHistory', JSON.stringify(newHistory));
      }
      
      setState({
        ...state,
        answers: newAnswers,
        showResults: true,
      });
    } else {
      setState({
        ...state,
        currentQuestionIndex: state.currentQuestionIndex + 1,
        answers: newAnswers,
      });
    }
    setSelectedAnswer(null);
    setShowFeedback(false);
  };

  const resetQuiz = () => {
    setState({
      currentQuestionIndex: 0,
      score: 0,
      answers: [],
      showResults: false,
    });
    setSelectedAnswer(null);
    setShowFeedback(false);
  };

  const saveScreenshot = async () => {
    if (resultsRef.current) {
      try {
        const dataUrl = await htmlToImage.toPng(resultsRef.current);
        const link = document.createElement('a');
        link.download = `quiz-results-${new Date().toISOString().slice(0, 10)}.png`;
        link.href = dataUrl;
        link.click();
      } catch (error) {
        console.error('Error saving screenshot:', error);
      }
    }
  };

  const getScoreMessage = (percentage: number) => {
    if (percentage >= 90) return "Excellent! You're an AI/ML expert! 🏆";
    if (percentage >= 70) return "Great job! You have solid knowledge! 🌟";
    if (percentage >= 50) return "Good effort! Keep learning! 📚";
    return "Keep studying! You'll improve! 💪";
  };

  if (state.showResults) {
    const percentage = (state.score / questions.length) * 100;
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div ref={resultsRef} className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold mb-4">Quiz Results</h2>
          <div className="mb-6">
            <p className="text-xl mb-2">Your score: {state.score} out of {questions.length} ({percentage.toFixed(1)}%)</p>
            <p className="text-lg text-blue-600">{getScoreMessage(percentage)}</p>
          </div>
          
          <div className="mb-6">
            <h3 className="text-xl font-semibold mb-3">Score History</h3>
            <div className="space-y-2">
              {scoreHistory.slice(-5).map((history, index) => (
                <div key={index} className="flex justify-between items-center border-b pb-2">
                  <span className="text-gray-600">{history.date}</span>
                  <span className="font-medium">{history.score}/{history.totalQuestions} ({history.percentage.toFixed(1)}%)</span>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            {questions.map((question, index) => (
              <div key={question.id} className="border rounded-lg p-4">
                <p className="font-semibold">{question.question}</p>
                <p className="text-green-600">Correct answer: {question.options[question.correctAnswer]}</p>
                {state.answers[index] !== question.correctAnswer && (
                  <div className="mt-2">
                    <p className="text-red-600">Your answer: {question.options[state.answers[index]]}</p>
                    <p className="mt-2">{question.explanation}</p>
                    <a href={question.reference} target="_blank" rel="noopener noreferrer" 
                       className="text-blue-600 hover:underline">
                      Learn more
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        <div className="mt-6 flex gap-4">
          <button
            onClick={resetQuiz}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
          <button
            onClick={saveScreenshot}
            className="flex-1 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Save Screenshot
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="mb-4">
        <span className="text-sm text-gray-600">
          Question {state.currentQuestionIndex + 1} of {questions.length}
        </span>
      </div>
      <div className="border rounded-lg p-6 mb-4">
        <h2 className="text-xl font-semibold mb-4">{currentQuestion.question}</h2>
        <div className="space-y-2">
          {currentQuestion.options.map((option, index) => (
            <button
              key={index}
              onClick={() => handleAnswer(index)}
              disabled={showFeedback}
              className={`w-full text-left p-3 rounded border transition-colors ${
                showFeedback
                  ? index === currentQuestion.correctAnswer
                    ? 'bg-green-100 border-green-500'
                    : index === selectedAnswer
                    ? 'bg-red-100 border-red-500'
                    : 'bg-gray-50'
                  : 'hover:bg-gray-50'
              }`}
            >
              {option}
            </button>
          ))}
        </div>
        {showFeedback && (
          <div className="mt-4 space-y-4">
            <div className={selectedAnswer === currentQuestion.correctAnswer ? 'text-green-600' : 'text-red-600'}>
              {selectedAnswer === currentQuestion.correctAnswer ? (
                <p>Correct! {currentQuestion.explanation}</p>
              ) : (
                <>
                  <p>Incorrect. {currentQuestion.explanation}</p>
                  <a
                    href={currentQuestion.reference}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline block mt-2"
                  >
                    Learn more
                  </a>
                </>
              )}
            </div>
            <button
              onClick={handleNextQuestion}
              className="w-full mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              {state.currentQuestionIndex === questions.length - 1 ? 'Show Results' : 'Next Question'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
} 