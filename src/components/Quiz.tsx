'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Question } from '../types/quiz';
import { questions } from '../data/quizData';
import FrequentlyMissed from './FrequentlyMissed';

interface ShuffledQuestion extends Question {
  shuffledOptions: string[];
  correctIndex: number;
}

interface QuizState {
  currentQuestionIndex: number;
  score: number;
  selectedAnswers: number[];
  showResults: boolean;
}

export const Quiz: React.FC = () => {
  const [isClient, setIsClient] = useState(false);
  const [state, setState] = useState<QuizState>({
    currentQuestionIndex: 0,
    score: 0,
    selectedAnswers: [],
    showResults: false
  });
  const [incorrectAnswers, setIncorrectAnswers] = useState<{ [key: number]: number }>({});
  const [activeTab, setActiveTab] = useState<'quiz' | 'history' | 'missed'>('quiz');
  const [scoreHistory, setScoreHistory] = useState<number[]>([]);
  const [quizQuestions, setQuizQuestions] = useState<ShuffledQuestion[]>([]);

  const startNewQuiz = useCallback(() => {
    // Shuffle questions and select first 15
    const shuffled = [...questions]
      .sort(() => Math.random() - 0.5)
      .slice(0, 15)
      .map(q => {
        const shuffledOptions = [...q.options].sort(() => Math.random() - 0.5);
        const correctIndex = shuffledOptions.indexOf(q.options[q.correctAnswer]);
        return {
          ...q,
          shuffledOptions,
          correctIndex
        };
      });
    setQuizQuestions(shuffled);
    setState({
      currentQuestionIndex: 0,
      score: 0,
      selectedAnswers: [],
      showResults: false
    });
  }, []);

  useEffect(() => {
    setIsClient(true);
    const savedHistory = localStorage.getItem('scoreHistory');
    if (savedHistory) {
      setScoreHistory(JSON.parse(savedHistory));
    }
    const savedIncorrect = localStorage.getItem('incorrectAnswers');
    if (savedIncorrect) {
      setIncorrectAnswers(JSON.parse(savedIncorrect));
    }
    startNewQuiz();
  }, [startNewQuiz]);

  const handleAnswerSelect = (answerIndex: number) => {
    const currentQuestion = quizQuestions[state.currentQuestionIndex];
    const isCorrect = answerIndex === currentQuestion.correctIndex;
    
    const newSelectedAnswers = [...state.selectedAnswers, answerIndex];
    const newScore = isCorrect ? state.score + 1 : state.score;
    
    if (!isCorrect) {
      setIncorrectAnswers(prev => {
        const newIncorrect = { ...prev, [currentQuestion.id]: (prev[currentQuestion.id] || 0) + 1 };
        if (isClient) {
          localStorage.setItem('incorrectAnswers', JSON.stringify(newIncorrect));
        }
        return newIncorrect;
      });
    }

    if (state.currentQuestionIndex < quizQuestions.length - 1) {
      setState(prev => ({
        ...prev,
        currentQuestionIndex: prev.currentQuestionIndex + 1,
        score: newScore,
        selectedAnswers: newSelectedAnswers
      }));
    } else {
      const finalScore = newScore + (isCorrect ? 1 : 0);
      const newScoreHistory = [...scoreHistory, finalScore];
      setScoreHistory(newScoreHistory);
      if (isClient) {
        localStorage.setItem('scoreHistory', JSON.stringify(newScoreHistory));
      }
      setState(prev => ({
        ...prev,
        score: finalScore,
        selectedAnswers: newSelectedAnswers,
        showResults: true
      }));
    }
  };

  const getFrequentlyMissedQuestions = () => {
    const missedQuestions = Object.entries(incorrectAnswers)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([id, count]) => {
        const question = questions.find(q => q.id === parseInt(id));
        return question ? { question, count } : null;
      })
      .filter((q): q is { question: Question; count: number } => q !== null);

    return missedQuestions;
  };

  if (!isClient) {
    return null;
  }

  if (activeTab === 'history') {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <h2 className="text-2xl font-bold mb-4">Score History</h2>
        <div className="space-y-2">
          {scoreHistory.map((score, index) => (
            <div key={index} className="p-4 bg-white rounded-lg shadow">
              Attempt {index + 1}: {score} correct answers
            </div>
          ))}
        </div>
        <button
          onClick={() => setActiveTab('quiz')}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Back to Quiz
        </button>
      </div>
    );
  }

  if (activeTab === 'missed') {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <FrequentlyMissed frequentlyMissed={getFrequentlyMissedQuestions()} />
        <button
          onClick={() => setActiveTab('quiz')}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Back to Quiz
        </button>
      </div>
    );
  }

  if (state.showResults) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <h2 className="text-2xl font-bold mb-4">Quiz Results</h2>
        <p className="text-lg mb-4">Your score: {state.score} out of {quizQuestions.length}</p>
        <div className="space-y-4">
          {quizQuestions.map((question, index) => (
            <div key={question.id} className="p-4 bg-white rounded-lg shadow">
              <p className="font-semibold">{question.question}</p>
              <p className={`mt-2 ${state.selectedAnswers[index] === question.correctIndex ? 'text-green-600' : 'text-red-600'}`}>
                Your answer: {question.shuffledOptions[state.selectedAnswers[index]]}
              </p>
              <p className="text-green-600">Correct answer: {question.options[question.correctAnswer]}</p>
              <p className="text-gray-600 mt-2">{question.explanation}</p>
              <a href={question.reference} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                Learn more
              </a>
            </div>
          ))}
        </div>
        <div className="mt-6 space-x-4">
          <button
            onClick={startNewQuiz}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Try Again
          </button>
          <button
            onClick={() => setActiveTab('missed')}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            View Missed Problems
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            View History
          </button>
        </div>
      </div>
    );
  }

  const currentQuestion = quizQuestions[state.currentQuestionIndex];

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="mb-4">
        <h2 className="text-2xl font-bold">Question {state.currentQuestionIndex + 1} of {quizQuestions.length}</h2>
        <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
          <div
            className="bg-blue-600 h-2.5 rounded-full"
            style={{ width: `${((state.currentQuestionIndex + 1) / quizQuestions.length) * 100}%` }}
          ></div>
        </div>
      </div>
      <div className="bg-white p-6 rounded-lg shadow">
        <p className="text-lg font-semibold mb-4">{currentQuestion.question}</p>
        <div className="space-y-3">
          {currentQuestion.shuffledOptions.map((option, index) => (
            <button
              key={index}
              onClick={() => handleAnswerSelect(index)}
              className="w-full p-3 text-left border rounded hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {option}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-6 space-x-4">
        <button
          onClick={() => setActiveTab('history')}
          className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          View History
        </button>
        <button
          onClick={() => setActiveTab('missed')}
          className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          View Missed Problems
        </button>
      </div>
    </div>
  );
}; 