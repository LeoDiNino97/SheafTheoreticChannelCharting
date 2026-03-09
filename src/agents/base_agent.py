from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAgent(nn.Module, ABC):
    """
    Abstract base class for all agents in the multi-agent training framework.

    This class defines the common interface that every agent must implement.
    Each agent is a PyTorch module and must provide implementations for
    the forward pass and its own loss computation.

    By inheriting from this class, derived agents can be handled uniformly
    by an external orchestrator responsible for coordinating training.

    Parameters
    ----------
    idx : int
        Unique identifier of the agent. This can be used by the orchestrator
        to differentiate agents during training, logging, or communication.

    Attributes
    ----------
    idx : int
        Unique identifier of the agent instance.
    """

    def __init__(
        self,
        idx: int,
        n: int
    ):
        """
        Initialize the base agent.

        Parameters
        ----------
        idx : int
            Unique identifier assigned to the agent.
        """
        super().__init__()

        self.idx = idx

        # Embedding dimension
        self.n = n

        # Keep the reference frame in the buffer for proper gradient flowing
        self.register_buffer("reference_frame", torch.eye(self.n))

    @abstractmethod
    def forward(
        self,
        xA: torch.Tensor,
        xP: torch.Tensor,
        xN: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the agent.

        This method defines how the agent processes an input tensor and
        produces its output. It must be implemented by all subclasses.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor provided to the agent.

        Returns
        -------
        torch.Tensor
            Output tensor produced by the agent.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        xA: torch.Tensor,
        xP: torch.Tensor,
        xN: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the training loss for the agent.

        Each agent is responsible for defining how its loss is computed.
        This allows different agents to optimize different objectives
        within the same training loop.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor used to compute the loss.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the loss value for this agent.
        """
        pass

    def reset_epoch_statistics(self):
        self.cov = torch.zeros((self.n,self.n), device=self.reference_frame.device)
    
    def accumulate_statistics(self, E, E_tilde, R_tilde):
        self.cov += E @ E_tilde.T @ R_tilde.T 

    def update_reference_frame(self):
        U, _, Vt = torch.linalg.svd(self.cov)
        with torch.no_grad():
            self.reference_frame.copy_(
                (
                    U @ 
                    torch.diag(
                        torch.cat(
                            [torch.ones(U.shape[1]-1),torch.tensor(torch.linalg.det(U @ Vt))]
                            )
                        ) @ 
                    Vt
                )
            )

