'use client';

import React, { useState, useRef, useEffect } from 'react';
import { QuizState, Question } from '../types/quiz';
import { questions } from '../data/quizData';
import * as htmlToImage from 'html-to-image';
import FrequentlyMissed from './FrequentlyMissed';

interface ScoreHistory {
  date: string;
  score: number;
  totalQuestions: number;
  percentage: number;
}

interface ShuffledOptions {
  options: string[];
  correctIndex: number;
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
  const [scoreHistory, setScoreHistory] = useState<ScoreHistory[]>([]);
  const [isClient, setIsClient] = useState(false);
  const [shuffledOptions, setShuffledOptions] = useState<ShuffledOptions[]>([]);
  const [incorrectAnswers, setIncorrectAnswers] = useState<{ [key: number]: number }>({});
  const [activeTab, setActiveTab] = useState<'quiz' | 'history' | 'missed'>('quiz');
  const [currentQuestions, setCurrentQuestions] = useState<Question[]>([]);
  
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsClient(true);
    const saved = localStorage.getItem('quizScoreHistory');
    if (saved) {
      setScoreHistory(JSON.parse(saved));
    }
    const storedIncorrect = localStorage.getItem('incorrectAnswers');
    if (storedIncorrect) {
      setIncorrectAnswers(JSON.parse(storedIncorrect));
    }
  }, []);

  const startNewQuiz = () => {
    // Shuffle all 70 questions and take the first 15
    const shuffledQuestions = [...questions]
      .sort(() => Math.random() - 0.5)
      .slice(0, 15);
    
    setCurrentQuestions(shuffledQuestions);
    setState(prev => ({
      ...prev,
      currentQuestionIndex: 0,
      score: 0,
      answers: [],
      showResults: false,
    }));
    setSelectedAnswer(null);
    setShowFeedback(false);
    setShuffledOptions(shuffledQuestions.map(question => ({
      options: [...question.options],
      correctIndex: question.correctAnswer
    })));
  };

  useEffect(() => {
    startNewQuiz();
  }, []);

  useEffect(() => {
    if (isClient) {
      localStorage.setItem('quizScoreHistory', JSON.stringify(scoreHistory));
      localStorage.setItem('incorrectAnswers', JSON.stringify(incorrectAnswers));
    }
  }, [scoreHistory, incorrectAnswers, isClient]);

  const currentQuestion = currentQuestions[state.currentQuestionIndex];
  const currentShuffled = shuffledOptions[state.currentQuestionIndex];

  const handleAnswer = (selectedOption: number) => {
    setSelectedAnswer(selectedOption);
    setShowFeedback(true);
    const isCorrect = selectedOption === currentShuffled.correctIndex;
    if (isCorrect) {
      setState(prev => ({
        ...prev,
        score: prev.score + 1,
      }));
    }

    if (!isCorrect) {
      setIncorrectAnswers(prev => ({
        ...prev,
        [currentQuestion.id]: (prev[currentQuestion.id] || 0) + 1
      }));
    }
  };

  const handleNextQuestion = () => {
    const newAnswers = [...state.answers, selectedAnswer!];
    
    if (state.currentQuestionIndex === currentQuestions.length - 1) {
      const newScore: ScoreHistory = {
        date: new Date().toLocaleString(),
        score: state.score,
        totalQuestions: currentQuestions.length,
        percentage: (state.score / currentQuestions.length) * 100
      };
      
      const newHistory = [...scoreHistory, newScore];
      setScoreHistory(newHistory);
      if (isClient) {
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

  const getFrequentlyMissed = () => {
    const missedQuestions = Object.entries(incorrectAnswers)
      .map(([id, count]) => {
        const question = questions.find(q => q.id === parseInt(id));
        return question ? { question, count } : null;
      })
      .filter((item): item is { question: Question; count: number } => item !== null)
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Show top 5 most frequently missed

    return missedQuestions;
  };

  if (state.showResults) {
    const percentage = (state.score / currentQuestions.length) * 100;
    return (
      <div className="max-w-3xl mx-auto">
        <div ref={resultsRef} className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
          <h2 className="text-3xl font-semibold text-gray-900 mb-6">Quiz Results</h2>
          <div className="mb-8">
            <p className="text-2xl mb-3 text-gray-900">Your score: {state.score} out of {currentQuestions.length}</p>
            <p className="text-4xl font-semibold text-blue-600 mb-4">{percentage.toFixed(1)}%</p>
            <p className="text-xl text-gray-600">{getScoreMessage(percentage)}</p>
          </div>
          
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Score History</h3>
            <div className="space-y-3">
              {scoreHistory.slice(-5).map((history, index) => (
                <div key={index} className="flex justify-between items-center border-b border-gray-100 pb-3">
                  <span className="text-gray-600">{history.date}</span>
                  <span className="font-medium text-gray-900">{history.score}/{history.totalQuestions} ({history.percentage.toFixed(1)}%)</span>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            {currentQuestions.map((question, index) => (
              <div key={question.id} className="border border-gray-100 rounded-xl p-6 hover:shadow-sm transition-shadow">
                <p className="font-semibold text-gray-900 mb-3">{question.question}</p>
                <p className="text-green-600 font-medium mb-3">Correct answer: {question.options[question.correctAnswer]}</p>
                {state.answers[index] !== shuffledOptions[index].correctIndex && (
                  <div className="mt-3">
                    <p className="text-red-600 font-medium mb-2">Your answer: {shuffledOptions[index].options[state.answers[index]]}</p>
                    <p className="text-gray-600 mb-2">{question.explanation}</p>
                    <a href={question.reference} target="_blank" rel="noopener noreferrer" 
                       className="text-blue-600 hover:text-blue-800 transition-colors">
                      Learn more
                    </a>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        <div className="mt-8 flex gap-4">
          <button
            onClick={startNewQuiz}
            className="flex-1 px-6 py-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors font-medium"
          >
            Try Again
          </button>
          <button
            onClick={saveScreenshot}
            className="flex-1 px-6 py-3 bg-gray-100 text-gray-900 rounded-full hover:bg-gray-200 transition-colors font-medium"
          >
            Save Results
          </button>
        </div>
      </div>
    );
  }

  if (!currentShuffled) {
    return <div>Loading...</div>;
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-8">
        <div className="flex space-x-2 border-b border-gray-200">
          <button
            onClick={() => setActiveTab('quiz')}
            className={`px-6 py-3 font-medium text-sm rounded-t-lg transition-colors ${
              activeTab === 'quiz'
                ? 'bg-white text-blue-600 border-t border-l border-r border-gray-200'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Quiz
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-6 py-3 font-medium text-sm rounded-t-lg transition-colors ${
              activeTab === 'history'
                ? 'bg-white text-blue-600 border-t border-l border-r border-gray-200'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            History
          </button>
          <button
            onClick={() => setActiveTab('missed')}
            className={`px-6 py-3 font-medium text-sm rounded-t-lg transition-colors ${
              activeTab === 'missed'
                ? 'bg-white text-blue-600 border-t border-l border-r border-gray-200'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Missed Problems
          </button>
        </div>
      </div>

      {activeTab === 'quiz' && (
        <>
          <div className="mb-6">
            <span className="text-sm text-gray-500">
              Question {state.currentQuestionIndex + 1} of {currentQuestions.length}
            </span>
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">{currentQuestion.question}</h2>
            <div className="space-y-3">
              {currentShuffled.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleAnswer(index)}
                  disabled={showFeedback}
                  className={`w-full text-left p-4 rounded-xl border transition-all duration-200 ${
                    showFeedback
                      ? index === currentShuffled.correctIndex
                        ? 'bg-green-50 border-green-200 text-green-900'
                        : index === selectedAnswer
                        ? 'bg-red-50 border-red-200 text-red-900'
                        : 'bg-gray-50 border-gray-100 text-gray-600'
                      : 'border-gray-200 hover:border-blue-200 hover:bg-blue-50 text-gray-900'
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
            {showFeedback && (
              <div className="mt-6 space-y-4">
                <div className={selectedAnswer === currentShuffled.correctIndex ? 'text-green-600' : 'text-red-600'}>
                  {selectedAnswer === currentShuffled.correctIndex ? (
                    <p className="text-lg">{currentQuestion.explanation}</p>
                  ) : (
                    <>
                      <p className="text-lg mb-2">{currentQuestion.explanation}</p>
                      <a
                        href={currentQuestion.reference}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 transition-colors inline-block"
                      >
                        Learn more
                      </a>
                    </>
                  )}
                </div>
                <button
                  onClick={handleNextQuestion}
                  className="w-full px-6 py-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors font-medium"
                >
                  {state.currentQuestionIndex === currentQuestions.length - 1 ? 'Show Results' : 'Next Question'}
                </button>
              </div>
            )}
          </div>
        </>
      )}

      {activeTab === 'history' && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6">Score History</h2>
          {scoreHistory.length === 0 ? (
            <p className="text-gray-500">No history yet</p>
          ) : (
            <div className="space-y-4">
              {scoreHistory.map((history, index) => (
                <div key={index} className="flex justify-between items-center border-b border-gray-100 pb-3">
                  <span className="text-gray-600">{history.date}</span>
                  <span className="font-medium text-gray-900">
                    {history.score}/{history.totalQuestions} ({history.percentage.toFixed(1)}%)
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'missed' && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
          <FrequentlyMissed frequentlyMissed={getFrequentlyMissed()} />
        </div>
      )}
    </div>
  );
} 