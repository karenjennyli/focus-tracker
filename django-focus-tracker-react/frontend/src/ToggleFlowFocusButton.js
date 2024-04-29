import { Box, Button } from "@chakra-ui/react";

const ToggleButtonGroup = ({ selectedButton, setSelectedButton }) => {
  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      p={1}
      bgColor="#2d3748"
      borderRadius="md"
    >
      <Button
        color={selectedButton === 'Flow' ? "white" : "gray.500"}
        backgroundColor={selectedButton === 'Flow' ? "#4173b4" : "#35507c"}
        width="60px"
        size="sm"
        onClick={() => setSelectedButton('Flow')}
        marginLeft={0.5}
        marginRight={0.5}
        _hover={{ backgroundColor: "#3f68a2" }}
      >
        Flow
      </Button>
      <Button
        color={selectedButton === 'Focus' ? "white" : "gray.500"}
        backgroundColor={selectedButton === 'Focus' ? "#4173b4" : "#35507c"}
        width="60px"
        size="sm"
        marginLeft={0.5}
        marginRight={0.5}
        onClick={() => setSelectedButton('Focus')}
        _hover={{ backgroundColor: "#3f68a2" }}
      >
        Focus
      </Button>
    </Box>
  );
};

export default ToggleButtonGroup;
